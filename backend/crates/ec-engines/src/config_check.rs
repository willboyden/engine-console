//! The Config files page's draft checks that need nothing of the hub (ADR 0023): the env file's (dotenv) and a node
//! policy's, with the diagnostics' scrubbing and cutting. Moved out of `ec-hub/src/services/config_check.rs`, which
//! re-exports them and keeps what reads the hub's files (GET /config/{file}) and the hosts.json check (its loader
//! reads the disk). The hub answers POST /config/validate with these functions, and the browser (`ec-engines-wasm`
//! `config_check`) runs the same ones on a draft without sending it: an env draft may hold a token.
//!
//! No diagnostic carries a value from the hub's own files or environment; a message may quote a key or a value of the
//! draft itself (a policy error names the key or path it refuses). Every message is scrubbed of secret-looking
//! tokens, each quoted echo is cut to [`MAX_ECHO_CHARS`], a message to [`MAX_MESSAGE_CHARS`], the list to
//! [`MAX_DIAGNOSTICS`], and the env check reports names only. No path a draft names is looked up on disk. An env
//! draft over [`MAX_ENV_DRAFT_LINES`] lines is refused before any check.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value, json};

use crate::envfile::{
    self, EnvSummary, LOOPBACK, MAX_ENV_EXPANDED, MAX_ENV_FILE, PREFIX, PREFIXED, SETTINGS,
    SettingsError, ValueNote,
};
use crate::policy_check;
use crate::text::scrub;

/// Settings that still parse but no longer do anything (docs/OPERATIONS.md "Settings that no longer do anything").
pub const NO_EFFECT: &[&str] = &[
    "hf_cache_dir",
    "docker_bin",
    "gateway_image",
    "gateway_image_allowlist",
    "discovery_enabled",
    "discovery_ports",
    "discovery_interval_s",
    "enable_api_docs",
];

/// The most diagnostics one answer lists (then one more line says how many were left out).
pub const MAX_DIAGNOSTICS: usize = 200;
/// A quoted echo of the draft ('...' or "...") in a message is cut to this many characters.
pub const MAX_ECHO_CHARS: usize = 80;
/// A whole message is cut to this many characters.
pub const MAX_MESSAGE_CHARS: usize = 500;
/// The longest env draft checked, in lines (a real env file has a few dozen); the hub's own reader has only the byte
/// cap, so a longer draft is refused here before any check runs.
pub const MAX_ENV_DRAFT_LINES: usize = 4000;

/// One message for the page: the text, and the 1-based line it is about when known.
pub fn diag(message: &str, line: Option<usize>, field: Option<&str>) -> Value {
    let mut m = Map::new();
    // scrub first: a cut secret-looking token might no longer look like one
    m.insert("message".into(), Value::String(shorten(&scrub(message))));
    if let Some(l) = line {
        m.insert("line".into(), Value::from(l));
    }
    if let Some(f) = field {
        m.insert("field".into(), Value::String(f.to_string()));
    }
    Value::Object(m)
}

/// `msg` with every quoted echo ('...' or "...", opened after a non-word character and closed before one) cut to
/// [`MAX_ECHO_CHARS`] characters plus "…", and the whole cut to [`MAX_MESSAGE_CHARS`]. Only the hub's answer is cut:
/// `policy_check` keeps the node's exact messages (the differential test compares them).
pub fn shorten(msg: &str) -> String {
    let chars: Vec<char> = msg.chars().collect();
    let n = chars.len();
    let word = |c: Option<&char>| c.is_some_and(|c| c.is_alphanumeric() || *c == '_');
    // Where each quote would close, found once, right to left (second audit F3: scanning forward from every opening
    // quote that never closes was O(k·n)). A closer scan starts just after a quote, which is never a backslash, so it
    // skips exactly the characters a scan from the start skips (each one after a backslash): mark those first.
    let mut escaped = vec![false; n];
    let mut j = 0;
    while j < n {
        if chars[j] == '\\' {
            if let Some(e) = escaped.get_mut(j + 1) {
                *e = true;
            }
            j += 2;
        } else {
            j += 1;
        }
    }
    let next_close = |q: char| {
        let mut next: Vec<Option<usize>> = vec![None; n + 1];
        for j in (0..n).rev() {
            next[j] = if !escaped[j] && chars[j] == q && !word(chars.get(j + 1)) {
                Some(j)
            } else {
                next[j + 1]
            };
        }
        next
    };
    let (single, double) = (next_close('\''), next_close('"'));
    let mut out = String::with_capacity(msg.len().min(MAX_MESSAGE_CHARS * 4));
    let mut i = 0;
    while i < n {
        let c = chars[i];
        let opens = (c == '\'' || c == '"') && !word(i.checked_sub(1).and_then(|j| chars.get(j)));
        let close = if !opens {
            None
        } else if c == '\'' {
            single[i + 1]
        } else {
            double[i + 1]
        };
        match close {
            Some(j) if j - i - 1 > MAX_ECHO_CHARS => {
                out.push(c);
                out.extend(&chars[i + 1..i + 1 + MAX_ECHO_CHARS]);
                out.push('…');
                out.push(c);
                i = j + 1;
            }
            Some(j) => {
                out.extend(&chars[i..=j]);
                i = j + 1;
            }
            None => {
                out.push(c);
                i += 1;
            }
        }
    }
    if out.chars().count() > MAX_MESSAGE_CHARS {
        out = out.chars().take(MAX_MESSAGE_CHARS).collect();
        out.push('…');
    }
    out
}

/// A name of the draft as a message shows it: at most [`MAX_ECHO_CHARS`] characters.
pub fn short_name(name: &str) -> String {
    if name.chars().count() > MAX_ECHO_CHARS {
        let mut s: String = name.chars().take(MAX_ECHO_CHARS).collect();
        s.push('…');
        s
    } else {
        name.to_string()
    }
}

static JSON_LINE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"at line (\d+) column \d+").ok());

/// The line a serde_json message points at ("... at line 3 column 7").
pub fn json_line(msg: &str) -> Option<usize> {
    JSON_LINE
        .as_ref()?
        .captures(msg)
        .and_then(|c| c[1].parse().ok())
}

/// The diagnostics, the outline (null when the draft does not load), as one check of a file gives them.
pub type Report = (Vec<Value>, Vec<Value>, Value);

/// The answer of POST /config/validate for `file` from one check's report: at most [`MAX_DIAGNOSTICS`] messages, then
/// one more line saying how many were left out.
pub fn finish(file: &str, report: Report) -> Value {
    let (mut errors, mut warnings, outline) = report;
    let total = errors.len() + warnings.len();
    if total > MAX_DIAGNOSTICS {
        errors.truncate(MAX_DIAGNOSTICS);
        warnings.truncate(MAX_DIAGNOSTICS - errors.len());
        let kept = errors.len() + warnings.len();
        warnings.push(diag(
            &format!(
                "{} more not shown (fix the ones above and validate again)",
                total - kept
            ),
            None,
            None,
        ));
    }
    json!({
        "file": file, "ok": errors.is_empty(), "errors": errors, "warnings": warnings, "outline": outline,
    })
}

/// The browser's POST /config/validate for the files it can check alone (`env`, `policy`): the same answer, with the
/// env draft read by [`envfile::check_env_draft`] (proven equal to the hub's loader). None for any other file.
pub fn validate_draft(file: &str, text: &str) -> Option<Value> {
    let report = match file {
        "env" => validate_env_with(text, envfile::check_env_draft),
        "policy" => validate_policy(text),
        _ => return None,
    };
    Some(finish(file, report))
}

/// Each assigned name and the 1-based line of its last assignment (the last one wins, as in the loader), in one pass.
pub fn env_lines(text: &str) -> HashMap<&str, usize> {
    let mut found = HashMap::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim_start();
        if line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").map_or(line, str::trim_start);
        if let Some((k, _)) = line.split_once('=') {
            found.insert(k.trim(), i + 1);
        }
    }
    found
}

/// The env check of a draft: `check` reads it as the hub would (the hub passes its loader, `config::check_env_text`;
/// the browser [`envfile::check_env_draft`]) and gives the names it sets and the first error or what the warnings need.
pub fn validate_env_with(
    text: &str,
    check: impl FnOnce(&str) -> (Vec<String>, Result<EnvSummary, SettingsError>),
) -> Report {
    if text.len() > MAX_ENV_FILE {
        let msg = format!(
            "over {} MiB: the hub would treat the whole file as unreadable and ignore it",
            MAX_ENV_FILE / (1024 * 1024)
        );
        return (vec![diag(&msg, None, None)], Vec::new(), Value::Null);
    }
    let lines_n = text.lines().count();
    if lines_n > MAX_ENV_DRAFT_LINES {
        let msg = format!(
            "{lines_n} lines: the editor checks env drafts of at most {MAX_ENV_DRAFT_LINES} lines (a real env file has a \
             few dozen settings)"
        );
        return (vec![diag(&msg, None, None)], Vec::new(), Value::Null);
    }
    if envfile::try_parse_dotenv(text, &[]).is_err() {
        let msg = format!(
            "the ${{VAR}} expansions add up to more than {} MiB: the hub would treat the whole file as unreadable and \
             ignore it",
            MAX_ENV_EXPANDED / (1024 * 1024)
        );
        return (vec![diag(&msg, None, None)], Vec::new(), Value::Null);
    }
    let lines = env_lines(text);
    let env_line = |name: &str| lines.get(name).copied();
    let (names, result) = check(text);
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut set: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    let pl = PREFIX.to_ascii_lowercase();
    for name in &names {
        let lk = name.to_ascii_lowercase();
        let line = env_line(name);
        let shown = short_name(name);
        if PREFIXED.contains(&lk.as_str()) {
            warnings.push(diag(
                &format!(
                    "{shown} is ignored: this setting is read only as {}",
                    envfile::env_var_name(&lk)
                ),
                line,
                Some(&lk),
            ));
            continue;
        }
        let field = match lk.strip_prefix(&pl) {
            Some(f) if PREFIXED.contains(&f) => f.to_string(),
            _ => lk.clone(),
        };
        if !SETTINGS.iter().any(|(n, _)| *n == field) {
            if unknown.len() < MAX_DIAGNOSTICS {
                unknown.push(shown.clone());
            }
            warnings.push(diag(
                &format!("{shown} is not a setting of the hub; it is ignored"),
                line,
                None,
            ));
            continue;
        }
        if NO_EFFECT.contains(&field.as_str()) {
            warnings.push(diag(
                &format!("{shown} still parses but no longer does anything"),
                line,
                Some(&field),
            ));
        }
        if field == "hf_token" {
            warnings.push(diag(
                "HF_TOKEN puts a secret in this file: prefer HF_TOKEN_FILE (a 0600 file of its own), and keep \
                 this file chmod 600",
                line,
                Some(&field),
            ));
        }
        set.push(envfile::env_var_name(&field));
    }
    for (name, field, note) in envfile::env_value_notes(text) {
        let shown = short_name(&name);
        let msg = match note {
            ValueNote::Tilde => format!(
                "{shown} starts with ~, which the hub never expands (it would be a directory named ~ under the \
                 directory the console starts in): write the absolute path"
            ),
            ValueNote::Negative => format!(
                "{shown} is negative, which the hub reads as 0: every request with a body on the paths this cap \
                 covers is then refused as too large"
            ),
        };
        warnings.push(diag(&msg, env_line(&name), Some(&field)));
    }
    if text.contains("${") {
        warnings.push(diag(
            "${VAR} expansions were checked with an empty environment (only their :-default applies); the hub \
             expands them from its own environment at start-up",
            None,
            None,
        ));
    }
    match &result {
        Ok(cfg) if cfg.trust_loopback => warnings.push(diag(
            "ENGINE_CONSOLE_TRUST_LOOPBACK is true: every local process and every page in a local browser is an \
             admin; keep it false",
            env_line("ENGINE_CONSOLE_TRUST_LOOPBACK"),
            Some("trust_loopback"),
        )),
        Ok(cfg) if !LOOPBACK.contains(&cfg.host.as_str()) => warnings.push(diag(
            "ENGINE_CONSOLE_HOST is not a loopback address: the console then listens beyond this machine; do that \
             only behind a TLS proxy, with keys",
            env_line("ENGINE_CONSOLE_HOST"),
            Some("host"),
        )),
        _ => {}
    }
    if let Err(e) = result {
        let var = envfile::env_var_name(&e.field);
        let line = names
            .iter()
            .rev()
            .find(|n| n.eq_ignore_ascii_case(&var))
            .and_then(|n| env_line(n));
        errors.push(diag(&format!("{var}: {}", e.message), line, Some(&e.field)));
    }
    (errors, warnings, json!({"set": set, "unknown": unknown}))
}

/// The policy check of a draft: the node's rules (`policy_check`, its mirror) on the text.
pub fn validate_policy(text: &str) -> Report {
    match policy_check::check_policy_text(text) {
        Ok(o) => {
            let mut warnings = vec![diag(
                "checked here without the node's own paths: the node also refuses a model root that overlaps its \
                 home, ~/.ssh or its own policy and state directories, and checks the file's owner and mode; run \
                 `ecnode check --policy <file>` on the machine for the final word",
                None,
                None,
            )];
            if o.mode == "monitor" && (o.allow_pull || o.allow_delete || o.writable_roots > 0) {
                warnings.push(diag(
                    "mode is monitor: allow_pull, allow_delete and writable roots take effect only in manage mode",
                    None,
                    Some("mode"),
                ));
            }
            let outline = json!({
                "mode": o.mode, "engine_images": o.engine_images, "model_roots": o.model_roots,
                "writable_roots": o.writable_roots, "scripts": o.scripts, "allow_pull": o.allow_pull,
                "allow_delete": o.allow_delete, "allow_ipc_host": o.allow_ipc_host,
                "external_control": o.external_control,
            });
            (Vec::new(), warnings, outline)
        }
        Err(msg) => {
            let line = json_line(&msg);
            (vec![diag(&msg, line, None)], Vec::new(), Value::Null)
        }
    }
}
