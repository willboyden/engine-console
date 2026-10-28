//! Python: node/src/ecnode/util.py.
//!
//! Shared helpers: subprocesses (`proc`: argv lists, never a shell, always a timeout, a child environment built
//! from scratch), Python text/path semantics (`py`), secret masking (`scrub`, `redact_argv`), secret-looking names
//! (`is_secret_name`), display-text cleaning and docker timestamps.

pub mod fsread;
pub mod panics;
pub mod proc;
pub mod py;
pub mod sync;

pub use proc::{
    CHILD_ENV_KEEP, CHILD_LANG, CmdResult, DEFAULT_PATH, EnvMap, Guarded, ProcRunner, RUN_MAX_ERR,
    RUN_MAX_OUT, Runner, TOOL_PATH_ENV, TOOL_PATH_OVERRIDE, child_env, iter_lines, kill_group,
    node_env, resolve_program, run, run_capped, run_tail, safe_path, spawn, tool_path,
    tool_path_with,
};
pub use sync::lock;

use std::sync::LazyLock;

use fancy_regex::{Captures, Regex};

pub const REDACTED: &str = "[redacted]";

/// Words that make a flag, key or variable name secret. "token" only as a whole word, so --tokenizer, max_tokens
/// and TOKENIZERS_PARALLELISM stay visible while --hf-token and HF_TOKEN are masked.
const SENSITIVE_WORD: &str = r"(?:api[-_]?key|apikey|secret|passw(?:or)?d|passphrase|credentials?|access[-_]?key|auth[-_]?key|private[-_]?key|(?<![a-z])token(?![a-z])|bearer|authorization)";

/// What a secret pattern becomes if it ever failed to compile: one that matches ANY text, so masking and
/// secret-name checks fail closed (everything masked, every name refused), never open. The unit tests assert that
/// no real pattern falls back to it.
const FAIL_CLOSED: &str = r"(?s)\A.*";

fn re(pattern: &str) -> Regex {
    // The patterns are constants: a failure here is a bug caught by the unit tests below.
    match Regex::new(pattern) {
        Ok(r) => r,
        Err(e) => unreachable_pattern(pattern, &e),
    }
}

#[cold]
fn unreachable_pattern(pattern: &str, e: &fancy_regex::Error) -> Regex {
    // Fall back to a pattern that matches everything rather than panicking in a node (fail closed).
    let _ = (pattern, e);
    Regex::new(FAIL_CLOSED).unwrap_or_else(|_| std::process::abort())
}

pub static SECRET_NAME_RE: LazyLock<Regex> = LazyLock::new(|| re(&format!("(?i){SENSITIVE_WORD}")));
static SECRET_VALUES: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{16,}|ec_[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[abprs]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}|nvapi-[A-Za-z0-9_\-]{20,}|eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})",
    )
});
static AUTH_HEADER: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?i)\b(Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=\-]{8,}"));
static URL_PASSWORD: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?i)([a-z][a-z0-9+.\-]*://[^/\s:@]{0,128}:)[^@\s/]{1,256}@"));
static KEY_VALUE: LazyLock<Regex> = LazyLock::new(|| {
    re(&format!(
        r#"(?i)([\w.\-]{{0,40}}{SENSITIVE_WORD}[\w.\-]{{0,40}}['"]?\s{{0,4}}[:=]\s{{0,4}}\[?\s{{0,4}}['"]?)([^'"\s,\]}}]{{1,4096}})"#
    ))
});
static FLAG_VALUE: LazyLock<Regex> = LazyLock::new(|| {
    re(&format!(
        r#"(?i)(--?[\w\-]{{0,40}}{SENSITIVE_WORD}[\w\-]{{0,40}})(\s+|=)(['"]?)([^\s'"]{{1,4096}})"#
    ))
});

/// `re.sub(f, text)`. fancy-regex can fail at MATCH time (its backtrack limit): then everything from the last
/// match on is replaced by one REDACTED, never passed through (fail closed).
fn sub(re: &Regex, text: &str, f: impl Fn(&Captures<'_, str>) -> String) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for caps in re.captures_iter(text) {
        let Ok(caps) = caps else {
            // backtrack limit hit: mask the rest rather than leak it
            out.push_str(REDACTED);
            return out;
        };
        let Some(m) = caps.get(0) else { continue };
        out.push_str(&text[last..m.start()]);
        out.push_str(&f(&caps));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

fn group<'t>(c: &Captures<'t, str>, i: usize) -> &'t str {
    c.get(i).map_or("", |m| m.as_str())
}

/// Mask secret-looking values in free text (log lines, argv tokens, error messages).
pub fn scrub(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let t = sub(&SECRET_VALUES, text, |_| REDACTED.to_string());
    let t = sub(&AUTH_HEADER, &t, |c| format!("{} {REDACTED}", group(c, 1)));
    let t = sub(&URL_PASSWORD, &t, |c| format!("{}{REDACTED}@", group(c, 1)));
    let t = sub(&FLAG_VALUE, &t, |c| {
        format!("{}{}{}{REDACTED}", group(c, 1), group(c, 2), group(c, 3))
    });
    sub(&KEY_VALUE, &t, |c| format!("{}{REDACTED}", group(c, 1)))
}

// Names a node never reads, even inside a model root (FLEET-PROTOCOL.md §4 "Path rules").
static SECRET_FILE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?is)^(?:id_.*|.*\.pem|.*\.key|.*\.p12|.*\.pfx|.*\.ppk|\.env.*|tokens?(?:[^a-z].*)?|credentials?(?:[^a-z].*)?|stored_tokens|\.netrc|\.git-credentials|\.pypirc|\.npmrc|authorized_keys.*|known_hosts.*)\n?\z",
    )
});
// Secret words anywhere in a name. A bare `token` substring is NOT one of them (tokenizer.json etc.).
static SECRET_WORD: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?i)secret|credential|passw(?:or)?d|api[-_.]?key|private[-_.]?key|(?:hf|api|access|auth|bearer|refresh|session|github|gh|hub|openai|ngc)[-_.]?tokens?(?![a-z])",
    )
});
pub const SECRET_DIRS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".aws",
    ".docker",
    ".kube",
    ".config",
    ".password-store",
    ".azure",
    ".gcloud",
    ".local",
];

/// `re.is_match(text)` where a match means "secret": a match-time error (fancy-regex's backtrack limit) counts as
/// a match, so a pattern that cannot decide refuses or masks (fail closed), never waves the text through.
fn hit(re: &Regex, text: &str) -> bool {
    re.is_match(text).unwrap_or(true)
}

/// True for a file or directory name that looks like a key, token or credential store (also when a pattern
/// cannot decide: fail closed).
pub fn is_secret_name(name: &str) -> bool {
    hit(&SECRET_FILE, name)
        || hit(&SECRET_WORD, name)
        || SECRET_DIRS.contains(&name.to_lowercase().as_str())
}

/// Names a hub may never CREATE or REPLACE inside a model root, besides every `is_secret_name` (security audit F2,
/// beyond Python; reads are unchanged): shell and session start-up files, which run at the next login, files that
/// hold credentials, and git's own directory and settings (hooks and config make git run commands). Compared
/// without regard to case.
pub const WRITE_PROTECTED_NAMES: &[&str] = &[
    ".bashrc",
    ".bash_profile",
    ".bash_login",
    ".bash_logout",
    ".bash_aliases",
    ".profile",
    ".zshrc",
    ".zshenv",
    ".zprofile",
    ".zlogin",
    ".zlogout",
    ".cshrc",
    ".tcshrc",
    ".login",
    ".logout",
    ".kshrc",
    ".xprofile",
    ".xsession",
    ".xsessionrc",
    ".xinitrc",
    ".pam_environment",
    ".pgpass",
    ".my.cnf",
    ".mylogin.cnf",
    ".gitconfig",
    ".git",
    ".netrc",
    ".git-credentials",
    "authorized_keys",
    "authorized_keys2",
];

/// True for a name `fs.write`, `fs.put_small` and `fs.symlink` refuse as any component of the path they write: a
/// secret-looking name (reads refuse those too) or one of WRITE_PROTECTED_NAMES.
pub fn is_write_protected_name(name: &str) -> bool {
    is_secret_name(name) || WRITE_PROTECTED_NAMES.contains(&name.to_lowercase().as_str())
}

/// True for a flag, key or variable name that looks secret (SECRET_NAME_RE; also when it cannot decide).
pub fn is_secret_flag(head: &str) -> bool {
    hit(&SECRET_NAME_RE, head)
}

fn secret_flag(head: &str) -> bool {
    is_secret_flag(head)
}

static CTRL: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"[\x00-\x1f\x7f]").unwrap_or_else(|_| std::process::abort())
});

/// util.py `redact_argv`: mask secret flag values BEFORE anything else looks at an argv: `--api-key X`,
/// `--token=X`, `NAME=X`, and the same inside one-string shell commands (`sh -c "vllm serve --api-key X"`).
pub fn redact_argv(argv: &[String], max_items: usize, max_len: usize) -> Vec<String> {
    redact_argv_depth(argv, max_items, max_len, 0)
}

fn redact_argv_depth(
    argv: &[String],
    max_items: usize,
    max_len: usize,
    depth: usize,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip = false;
    for raw in argv.iter().take(max_items) {
        let replaced = CTRL.replace_all(raw, " ");
        let tok = py::take(&replaced, max_len).to_string();
        if skip {
            out.push(REDACTED.to_string());
            skip = false;
            continue;
        }
        if depth < 2 && py::strip(&tok).chars().any(py::isspace) {
            let parts = match py::shlex_split(&tok) {
                Ok(p) => p,
                Err(_) => py::split_ws(&tok).into_iter().map(str::to_string).collect(),
            };
            out.push(redact_argv_depth(&parts, max_items, max_len, depth + 1).join(" "));
            continue;
        }
        let (head, eq) = match tok.split_once('=') {
            Some((h, _)) => (h, true),
            None => (tok.as_str(), false),
        };
        if secret_flag(head) && (head.starts_with('-') || eq) {
            if eq {
                out.push(format!("{head}={REDACTED}"));
            } else {
                out.push(tok.clone());
                skip = true;
            }
            continue;
        }
        out.push(scrub(&tok));
    }
    out
}

/// util.py `json_size`: bytes `v` takes in a protocol frame (compact JSON, UTF-8).
pub fn json_size(v: &serde_json::Value) -> usize {
    serde_json::to_vec(v).map_or(usize::MAX, |b| b.len())
}

fn str_json_size(s: &str) -> usize {
    json_size(&serde_json::Value::String(s.to_string()))
}

/// util.py `fit_tail`: the longest tail of `text` whose JSON encoding fits in `budget` bytes. (tail, trimmed).
pub fn fit_tail(text: &str, budget: usize) -> (String, bool) {
    if str_json_size(text) <= budget {
        return (text.to_string(), false);
    }
    let idx: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let n = idx.len();
    let at = |k: usize| if k >= n { text.len() } else { idx[k] };
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if str_json_size(&text[at(mid)..]) <= budget {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    (text[at(lo)..].to_string(), true)
}

/// util.py `clean_text`: one display line, control characters removed, capped (characters).
pub fn clean_text(s: &str, limit: usize) -> String {
    s.chars()
        .filter(|c| !matches!(*c, '\u{0}'..='\u{1f}' | '\u{7f}'))
        .take(limit)
        .collect()
}

/// `clean_text(str(v), limit)` of a JSON value.
pub fn clean_value(v: &serde_json::Value, limit: usize) -> String {
    clean_text(&py::str_of(v), limit)
}

/// util.py `as_int`
pub fn as_int(v: &serde_json::Value) -> Option<i64> {
    match v {
        serde_json::Value::Number(n) if !n.is_f64() => n.as_i64(),
        serde_json::Value::String(s) => py::parse_int(s),
        _ => None,
    }
}

/// util.py `as_float`
pub fn as_float(v: &serde_json::Value) -> Option<f64> {
    let f = match v {
        serde_json::Value::Number(n) => n.as_f64()?,
        serde_json::Value::String(s) => py::parse_float(s)?,
        _ => return None,
    };
    f.is_finite().then_some(f)
}

static TS: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r"^([0-9]{4}-[0-9]{2}-[0-9]{2})[T ]([0-9]{2}:[0-9]{2}:[0-9]{2})(\.[0-9]+)?\s*(Z|[+-][0-9]{2}:?[0-9]{2})?",
    )
    .unwrap_or_else(|_| std::process::abort())
});

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        _ => 28,
    }
}

/// util.py `parse_time`: an RFC 3339 timestamp as docker prints it (nanoseconds, 'Z') -> epoch seconds; None for
/// docker's zero time or anything unparsable.
pub fn parse_time(v: &serde_json::Value) -> Option<f64> {
    let serde_json::Value::String(s) = v else {
        return None;
    };
    if s.is_empty() || s.starts_with("0001-01-01") {
        return None;
    }
    let c = TS.captures(py::strip(s))?;
    let date = c.get(1)?.as_str();
    let time = c.get(2)?.as_str();
    let num = |t: &str| t.parse::<i64>().ok();
    let (y, mo, d) = (num(&date[0..4])?, num(&date[5..7])?, num(&date[8..10])?);
    let (h, mi, se) = (num(&time[0..2])?, num(&time[3..5])?, num(&time[6..8])?);
    if !(1..=9999).contains(&y)
        || !(1..=12).contains(&mo)
        || d < 1
        || d > days_in_month(y, mo)
        || h > 23
        || mi > 59
        || se > 59
    {
        return None;
    }
    let base = (days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + se) as f64;
    let mut offset = 0i64;
    if let Some(tz) = c.get(4).map(|m| m.as_str())
        && tz != "Z"
    {
        let sign = if tz.starts_with('-') { -1 } else { 1 };
        let hh = num(&tz[1..3])?;
        let mm = num(&tz[tz.len() - 2..])?;
        offset = sign * (hh * 3600 + mm * 60);
    }
    let frac = match c.get(3) {
        Some(m) => format!("0{}", m.as_str()).parse::<f64>().ok()?,
        None => 0.0,
    };
    Some(base - offset as f64 + frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_compile() {
        for r in [
            &*SECRET_NAME_RE,
            &*SECRET_VALUES,
            &*AUTH_HEADER,
            &*URL_PASSWORD,
            &*KEY_VALUE,
            &*FLAG_VALUE,
            &*SECRET_FILE,
            &*SECRET_WORD,
        ] {
            assert_ne!(r.as_str(), FAIL_CLOSED);
        }
        // the fallback itself matches anything, the empty name included
        let f = re("(");
        assert_eq!(f.as_str(), FAIL_CLOSED);
        assert!(hit(&f, "") && hit(&f, "config.json"));
        assert_eq!(sub(&f, "plain text", |_| REDACTED.to_string()), REDACTED);
    }

    /// A matcher that fails at match time (fancy-regex's backtrack limit) must count as "secret": masked text,
    /// refused names. Injected: a real pattern with a backtrack limit too small for this input.
    #[test]
    fn a_match_time_regex_error_fails_closed() {
        let failing = fancy_regex::RegexBuilder::new(&format!("(?i){SENSITIVE_WORD}"))
            .backtrack_limit(1)
            .build()
            .unwrap_or_else(|e| panic!("{e}"));
        let name = "my-token";
        assert!(
            failing.is_match(name).is_err(),
            "the injected matcher must fail"
        );
        assert!(hit(&failing, name));
        let out = sub(&failing, "keep --token SUPERSECRETVALUE1234 rest", |_| {
            "X".to_string()
        });
        assert!(!out.contains("SUPERSECRET"), "{out}");
        assert!(out.ends_with(REDACTED), "{out}");
    }

    /// The real patterns on pathological input: whatever the regex engine does (match, give up), no secret value
    /// comes out and names stay refused.
    #[test]
    fn pathological_inputs_never_leak() {
        let secret = "VALUEZQ9VLMT4PR2SXW8K";
        for text in [
            format!("{} --api-key {secret}", "a-".repeat(50_000)),
            format!("{} token={secret}", "x".repeat(200_000)),
            format!("{}\"secret\": \"{secret}\"", "_.-".repeat(30_000)),
            format!(
                "{} Authorization: Bearer {secret}",
                "tokentoken ".repeat(20_000)
            ),
        ] {
            let out = scrub(&text);
            assert!(!out.contains(secret), "leaked in {} chars", out.len());
        }
        let long_name = format!("{}-api-key", "a".repeat(100_000));
        assert!(is_secret_name(&long_name));
    }

    #[test]
    fn scrub_masks_the_python_cases() {
        assert_eq!(
            scrub("--api-key sk-abcdefghijklmnopqrstu"),
            "--api-key [redacted]"
        );
        assert_eq!(
            scrub("Authorization: Bearer abcdefghijkl"),
            "Authorization: [redacted] [redacted]"
        );
        assert_eq!(
            scrub("http://user:pw@proxy.invalid:3128"),
            "http://user:[redacted]@proxy.invalid:3128"
        );
        assert_eq!(
            scrub("--tokenizer foo max_tokens=5"),
            "--tokenizer foo max_tokens=5"
        );
        assert_eq!(scrub("HF_TOKEN=abc123"), "HF_TOKEN=[redacted]");
        assert_eq!(scrub("hf_abcdefghijklmnopqrstuvwxyz"), "[redacted]");
    }

    #[test]
    fn secret_names() {
        for n in [
            "id_rsa",
            "x.pem",
            ".env.local",
            "token",
            "tokens.json",
            "hf_token",
            "my-secret.txt",
            ".ssh",
            ".SSH",
        ] {
            assert!(is_secret_name(n), "{n}");
        }
        for n in [
            "tokenizer.json",
            "special_tokens_map.json",
            "config.json",
            "model.safetensors",
        ] {
            assert!(!is_secret_name(n), "{n}");
        }
    }
}
