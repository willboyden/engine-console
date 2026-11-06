//! The Config files page: draft, validate, export. Nothing here writes, renames or chmods a file, and nothing runs a
//! command: the hub keeps its config files out of the browser's reach on purpose (hosts.json decides what the hub
//! runs ssh against; the env file may hold HF_TOKEN; a node policy lives on its own machine and only its owner may
//! change it). The page drafts a file in the browser, this module checks the draft with the loaders the hub and the
//! node really use, and the human installs the result with the commands the page shows.
//!
//! * [`current`]: what the page may load. hosts.json is served read-only, through the loader's own file checks, only
//!   when the loader accepts it and it holds nothing secret-looking (its schema has no secret field). The env file is
//!   never served (it may hold a token): only its path and whether it exists. A node policy is not on this machine.
//!   Each answer carries the repository's template(s) for that file.
//! * [`validate`]: the real checks on a draft (`hosts::parse_hosts_bytes`, `config::check_env_text`,
//!   `policy_check::check_policy_text`, the node validator's mirror). No diagnostic carries a value from the hub's
//!   own files or environment; a message may quote a key or a value of the draft itself (a hosts or policy error
//!   names the key or path it refuses). Every message is scrubbed of secret-looking tokens, each quoted echo is cut
//!   to [`MAX_ECHO_CHARS`], a message to [`MAX_MESSAGE_CHARS`], the list to [`MAX_DIAGNOSTICS`], and the env check
//!   reports names only. No path a draft names is looked up on disk (the env check leaves out start-up's
//!   `HF_CACHE_DIR` symlink test). An env draft over [`MAX_ENV_DRAFT_LINES`] lines is refused before any check.

use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value, json};

use crate::config::{self, MAX_ENV_FILE, PREFIX, PREFIXED, SETTINGS, Settings, ValueNote};
use crate::services::common::scrub;
use crate::services::hosts::{self, MAX_FILE_BYTES as MAX_HOSTS_BYTES};
use crate::services::policy_check::{self, MAX_POLICY_BYTES};

/// The files the page knows, by id.
pub const FILES: &[&str] = &["hosts", "env", "policy"];

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

/// The node looks for its policy here (`ecnode::policy`: `--policy PATH`, else the first, else the second).
pub const POLICY_LOCATIONS: &[&str] = &["/etc/ecnode/policy.json", "~/.config/ecnode/policy.json"];

pub const HOSTS_TEMPLATE: &str = r#"{"hosts": [
  {"id": "local", "label": "This machine"},
  {"id": "gpu-box-2", "label": "GPU box 2", "ssh": "ops@192.0.2.5"},
  {"id": "edge-1", "label": "Edge 1", "ssh": "ops@192.0.2.6", "port": 2222, "enabled": false}
]}
"#;

pub const ENV_TEMPLATE: &str = "# Engine Console hub settings (dotenv: NAME=value, one per line). Real environment variables win over
# this file. The hub reads it once at start-up: restart the console after a change. Keep it chmod 600.
# Remove the # in front of a line to use it.

# Hugging Face traffic through your egress proxy (with its CA certificate, never its key)
# EGRESS_PROXY=http://127.0.0.1:8082
# EGRESS_CA_BUNDLE=/etc/ssl/certs/egress-proxy-ca.pem
# REQUIRE_EGRESS_PROXY=true

# A token for gated models: keep it in its own 0600 file rather than here
# (an absolute path: the hub never expands ~)
# HF_TOKEN_FILE=/path/to/engine-console/hf_token

# Where the console listens; widen only behind a TLS proxy with keys
# ENGINE_CONSOLE_HOST=127.0.0.1
# ENGINE_CONSOLE_PORT=8791

# Live streams at once, in total and per API key
# MAX_SSE_CONNECTIONS=8
# MAX_SSE_PER_KEY=4
";

pub const POLICY_MONITOR: &str = include_str!("../../../../../node/examples/policy.monitor.json");
pub const POLICY_MANAGE: &str = include_str!("../../../../../node/examples/policy.manage.json");

/// One message for the page: the text, and the 1-based line it is about when known.
fn diag(message: &str, line: Option<usize>, field: Option<&str>) -> Value {
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
fn short_name(name: &str) -> String {
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
fn json_line(msg: &str) -> Option<usize> {
    JSON_LINE
        .as_ref()?
        .captures(msg)
        .and_then(|c| c[1].parse().ok())
}

/// GET /config/{file}: what the page may load for `file` (None: not a file the page knows).
pub fn current(file: &str, cfg: &Settings) -> Option<Value> {
    current_as(
        file,
        cfg,
        &config::env_file(),
        nix::unistd::geteuid().as_raw(),
    )
}

/// `current` with the env file path and the user given (tests).
pub fn current_as(file: &str, cfg: &Settings, env_path: &Path, euid: u32) -> Option<Value> {
    Some(match file {
        "hosts" => hosts_current(&cfg.hosts_file, euid),
        "env" => json!({
            "file": "env", "format": "dotenv", "path": env_path.display().to_string(),
            "exists": env_path.symlink_metadata().is_ok(), "served": false, "text": null,
            "reason": "secret", "problem": null,
            "max_bytes": MAX_ENV_FILE, "max_lines": MAX_ENV_DRAFT_LINES,
            "templates": {"env": ENV_TEMPLATE},
            "fields": SETTINGS.iter().map(|(n, k)| json!({"name": n, "var": config::env_var_name(n), "kind": k,
                "no_effect": NO_EFFECT.contains(n)})).collect::<Vec<_>>(),
        }),
        "policy" => json!({
            "file": "policy", "format": "json", "path": null, "locations": POLICY_LOCATIONS,
            "node_policy": cfg.node_policy.as_ref().map(|p| p.display().to_string()),
            "exists": null, "served": false, "text": null, "reason": "remote", "problem": null,
            "max_bytes": MAX_POLICY_BYTES,
            "templates": {"monitor": POLICY_MONITOR, "manage": POLICY_MANAGE},
        }),
        _ => return None,
    })
}

fn hosts_current(path: &Path, euid: u32) -> Value {
    let mut out = json!({
        "file": "hosts", "format": "json", "path": path.display().to_string(), "exists": false, "served": false,
        "text": null, "reason": null, "problem": null, "max_bytes": MAX_HOSTS_BYTES,
        "templates": {"hosts": HOSTS_TEMPLATE},
    });
    match hosts::read_hosts_file_as(path, euid) {
        Ok(None) => out["reason"] = json!("missing"),
        Err(e) => {
            out["exists"] = json!(path.symlink_metadata().is_ok());
            out["reason"] = json!("refused");
            out["problem"] = json!(scrub(&e.to_string()));
        }
        Ok(Some(data)) => {
            out["exists"] = json!(true);
            match (hosts::parse_hosts_bytes(&data), String::from_utf8(data)) {
                (Ok(_), Ok(text)) if scrub(&text) == text => {
                    out["served"] = json!(true);
                    out["text"] = json!(text);
                }
                (Ok(_), Ok(_)) => out["reason"] = json!("secret"),
                (Err(e), _) => {
                    out["reason"] = json!("invalid");
                    out["problem"] = json!(scrub(&format!("{}: {e}", path.display())));
                }
                (Ok(_), Err(_)) => out["reason"] = json!("invalid"),
            }
        }
    }
    out
}

/// POST /config/validate: the real checks on a draft of `file`. Never writes (the env check stats HF_CACHE_DIR, as
/// the hub's own validator does). None: not a file the page knows.
pub fn validate(file: &str, text: &str) -> Option<Value> {
    let (mut errors, mut warnings, outline) = match file {
        "hosts" => validate_hosts(text),
        "env" => validate_env(text),
        "policy" => validate_policy(text),
        _ => return None,
    };
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
    Some(json!({
        "file": file, "ok": errors.is_empty(), "errors": errors, "warnings": warnings, "outline": outline,
    }))
}

type Report = (Vec<Value>, Vec<Value>, Value);

fn validate_hosts(text: &str) -> Report {
    match hosts::parse_hosts_bytes(text.as_bytes()) {
        Ok(list) => {
            let mut warnings = Vec::new();
            if !list.iter().any(|h| h.id == crate::domain::fleet::LOCAL) {
                warnings.push(diag(
                    "no host \"local\": the console then manages only other machines, and API calls that omit \
                     host answer 404 unknown_host",
                    None,
                    None,
                ));
            }
            if list.iter().any(|h| h.ssh_config) {
                warnings.push(diag(
                    "ssh_config true lets your ssh configuration add identity files and options; leave it off \
                     unless you need an alias or a ProxyJump",
                    None,
                    Some("ssh_config"),
                ));
            }
            let hosts: Vec<Value> = list
                .iter()
                .map(|h| {
                    json!({"id": h.id, "label": h.label, "remote": h.ssh.is_some(), "enabled": h.enabled})
                })
                .collect();
            (Vec::new(), warnings, json!({"hosts": hosts}))
        }
        Err(e) => {
            let msg = e.to_string();
            let line = json_line(&msg);
            (vec![diag(&msg, line, None)], Vec::new(), Value::Null)
        }
    }
}

/// Each assigned name and the 1-based line of its last assignment (the last one wins, as in the loader), in one pass.
fn env_lines(text: &str) -> HashMap<&str, usize> {
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

fn validate_env(text: &str) -> Report {
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
    if config::try_parse_dotenv(text, &[]).is_err() {
        let msg = format!(
            "the ${{VAR}} expansions add up to more than {} MiB: the hub would treat the whole file as unreadable and \
             ignore it",
            config::MAX_ENV_EXPANDED / (1024 * 1024)
        );
        return (vec![diag(&msg, None, None)], Vec::new(), Value::Null);
    }
    let lines = env_lines(text);
    let env_line = |name: &str| lines.get(name).copied();
    let (names, result) = config::check_env_text(text);
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
                    config::env_var_name(&lk)
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
        set.push(config::env_var_name(&field));
    }
    for (name, field, note) in config::env_value_notes(text) {
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
        Ok(cfg) if !crate::api::security::LOOPBACK.contains(&cfg.host.as_str()) => warnings.push(diag(
            "ENGINE_CONSOLE_HOST is not a loopback address: the console then listens beyond this machine; do that \
             only behind a TLS proxy, with keys",
            env_line("ENGINE_CONSOLE_HOST"),
            Some("host"),
        )),
        _ => {}
    }
    if let Err(e) = result {
        let var = config::env_var_name(&e.field);
        let line = names
            .iter()
            .rev()
            .find(|n| n.eq_ignore_ascii_case(&var))
            .and_then(|n| env_line(n));
        errors.push(diag(&format!("{var}: {}", e.message), line, Some(&e.field)));
    }
    (errors, warnings, json!({"set": set, "unknown": unknown}))
}

fn validate_policy(text: &str) -> Report {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_lines_finds_the_last_assignment() {
        let t = "# A=0\nA=1\nexport B=2\nA=3\n";
        let l = env_lines(t);
        assert_eq!(l.get("A"), Some(&4));
        assert_eq!(l.get("B"), Some(&3));
        assert_eq!(l.get("C"), None);
    }

    #[test]
    fn test_shorten_cuts_quoted_echoes_and_long_messages() {
        let long = "x".repeat(300);
        let got = shorten(&format!("model_roots[0].path: '/{long}' is too broad"));
        assert_eq!(
            got,
            format!("model_roots[0].path: '/{}…' is too broad", "x".repeat(79))
        );
        // an apostrophe inside a word opens nothing; short quotes are kept
        assert_eq!(
            shorten("the node's rules: 'abc' and \"d'e\""),
            "the node's rules: 'abc' and \"d'e\""
        );
        let got = shorten(&"y ".repeat(1000));
        assert_eq!(got.chars().count(), MAX_MESSAGE_CHARS + 1);
        assert!(got.ends_with('…'));
    }

    /// The forward-scanning `shorten` the linear one replaced (second audit F3), kept as the reference it must equal.
    fn shorten_reference(msg: &str) -> String {
        let chars: Vec<char> = msg.chars().collect();
        let word = |c: Option<&char>| c.is_some_and(|c| c.is_alphanumeric() || *c == '_');
        let mut out = String::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            let opens =
                (c == '\'' || c == '"') && !word(i.checked_sub(1).and_then(|j| chars.get(j)));
            let mut close = None;
            if opens {
                let mut j = i + 1;
                while j < chars.len() {
                    if chars[j] == '\\' {
                        j += 2;
                        continue;
                    }
                    if chars[j] == c && !word(chars.get(j + 1)) {
                        close = Some(j);
                        break;
                    }
                    j += 1;
                }
            }
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

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(2000))]
        #[test]
        fn test_shorten_equals_the_forward_scan(msg in "[ab_ '\"\\\\:,é]{0,300}") {
            proptest::prop_assert_eq!(shorten(&msg), shorten_reference(&msg));
        }
    }

    #[test]
    fn test_shorten_long_echoes_equal_the_forward_scan() {
        for msg in [
            format!("a '{}' b \"{}\" c", "x\\'y ".repeat(40), "z\\\"".repeat(50)),
            format!("'{}", "q' ".repeat(200)),
            format!("k: '{}'", "\\\\".repeat(100)),
        ] {
            assert_eq!(shorten(&msg), shorten_reference(&msg), "{msg}");
        }
    }

    #[test]
    fn test_shorten_is_linear_on_unclosed_quotes() {
        // second audit F3: 64 KiB of apostrophe-led fragments, none closed, took O(k·n)
        let msg = " 'a,".repeat(16 * 1024);
        let t0 = std::time::Instant::now();
        let got = shorten(&msg);
        assert!(
            t0.elapsed() < std::time::Duration::from_millis(500),
            "{:?}",
            t0.elapsed()
        );
        assert_eq!(got.chars().count(), MAX_MESSAGE_CHARS + 1);
    }

    #[test]
    fn test_json_line() {
        assert_eq!(json_line("expected `,` at line 3 column 7"), Some(3));
        assert_eq!(json_line("no position"), None);
    }
}
