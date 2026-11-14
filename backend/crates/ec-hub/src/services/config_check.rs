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
//!   `policy_check::check_policy_text`, the node validator's mirror). The env and policy checks and the cutting below
//!   live in `ec_engines::config_check` (ADR 0023), which the browser also runs. No diagnostic carries a value from the hub's
//!   own files or environment; a message may quote a key or a value of the draft itself (a hosts or policy error
//!   names the key or path it refuses). Every message is scrubbed of secret-looking tokens, each quoted echo is cut
//!   to [`MAX_ECHO_CHARS`], a message to [`MAX_MESSAGE_CHARS`], the list to [`MAX_DIAGNOSTICS`], and the env check
//!   reports names only. No path a draft names is looked up on disk (the env check leaves out start-up's
//!   `HF_CACHE_DIR` symlink test). An env draft over [`MAX_ENV_DRAFT_LINES`] lines is refused before any check.

use std::path::Path;

use serde_json::{Value, json};

// The env and policy checks and the diagnostics' scrubbing and cutting are shared with the browser (ADR 0023).
pub use ec_engines::config_check::{
    MAX_DIAGNOSTICS, MAX_ECHO_CHARS, MAX_ENV_DRAFT_LINES, MAX_MESSAGE_CHARS, NO_EFFECT, Report,
    shorten,
};
use ec_engines::config_check::{diag, finish, json_line, validate_env_with, validate_policy};
use ec_engines::envfile::EnvSummary;

use crate::config::{self, MAX_ENV_FILE, SETTINGS, Settings};
use crate::services::common::scrub;
use crate::services::hosts::{self, MAX_FILE_BYTES as MAX_HOSTS_BYTES};
use crate::services::policy_check::MAX_POLICY_BYTES;

/// The files the page knows, by id.
pub const FILES: &[&str] = &["hosts", "env", "policy"];

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

/// POST /config/validate: the real checks on a draft of `file`. Never writes, and looks up no path the draft names.
/// The env and policy checks are `ec_engines::config_check`'s (the browser runs the same on a draft it does not send);
/// the env draft is read by the hub's own loader here. None: not a file the page knows.
pub fn validate(file: &str, text: &str) -> Option<Value> {
    let report = match file {
        "hosts" => validate_hosts(text),
        "env" => validate_env_with(text, |t| {
            let (names, r) = config::check_env_text(t);
            (
                names,
                r.map(|s| EnvSummary {
                    host: s.host,
                    trust_loopback: s.trust_loopback,
                }),
            )
        }),
        "policy" => validate_policy(text),
        _ => return None,
    };
    Some(finish(file, report))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use ec_engines::config_check::env_lines;

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
