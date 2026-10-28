//! `engine-console hosts ...` (port of cli.py): the command line; the work is in services/hosts.rs and
//! services/fleet.rs.
//!
//!   hosts init-key --from ADDRS [--key-file PATH]   create the console's own ssh key and print the one line each
//!                                                   machine must authorize (install-nodes.sh parses that line)
//!   hosts check [--timeout S]                       connect to every host once and report what each node says
//!   hosts list                                      the configured hosts, without connecting
//!
//! Exit codes as argparse/cli.py: 0 ok, 1 a refused setting or a host that is not up, 2 a usage error.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::Settings;
use crate::domain::fleet::{Fleet, HostSummary};
use crate::services::common::EventBus;
use crate::services::fleet::FleetService;
use crate::services::hosts::{HostsError, init_key, load_hosts};

pub const CHECK_TIMEOUT_S: f64 = 20.0;
/// a link in any of these has finished its first attempt
pub const SETTLED: &[&str] = &["up", "stale", "down", "incompatible", "disabled"];
const PROG: &str = "engine-console hosts";
const USAGE: &str = "usage: engine-console hosts [-h] {init-key,check,list} ...";

// The help texts are argparse's output at its default width (80 columns: COLUMNS unset and stdout not a terminal),
// byte for byte (tests/test_hosts_cli_golden.rs). argparse re-wraps for COLUMNS or the terminal's width; these do
// not (PORTING.md §14 DV14).
const HELP: &str = "usage: engine-console hosts [-h] {init-key,check,list} ...

Set up and check the machines the console manages.

positional arguments:
  {init-key,check,list}
    init-key            create the console's ssh key (if missing) and print
                        the authorized_keys line
    check               connect to every host once and print its state,
                        versions and policy mode
    list                list the configured hosts without connecting

options:
  -h, --help            show this help message and exit
";
const HELP_INIT_KEY: &str =
    "usage: engine-console hosts init-key [-h] --from ADDRS [--key-file KEY_FILE]

options:
  -h, --help           show this help message and exit
  --from ADDRS         the console machine's address as the others see it
                       (comma-separated, e.g. 10.0.0.5): each machine then
                       accepts the key from there only
  --key-file KEY_FILE  where the key lives (default: the HUB_KEY_FILE setting)
";
const HELP_CHECK: &str = "usage: engine-console hosts check [-h] [--timeout TIMEOUT]

options:
  -h, --help         show this help message and exit
  --timeout TIMEOUT  seconds to wait for every host (default 20)
";
const HELP_LIST: &str = "usage: engine-console hosts list [-h]

options:
  -h, --help  show this help message and exit
";

#[derive(Debug, Clone, PartialEq)]
enum Cmd {
    InitKey {
        from: String,
        key_file: Option<PathBuf>,
    },
    Check {
        timeout: f64,
    },
    List,
    Help(&'static str),
}

/// argparse's usage error: message on stderr, exit 2.
fn usage_error(sub: Option<&str>, msg: &str) -> i32 {
    let usage = match sub {
        Some("init-key") => {
            format!("usage: {PROG} init-key [-h] --from ADDRS [--key-file KEY_FILE]")
        }
        Some("check") => format!("usage: {PROG} check [-h] [--timeout TIMEOUT]"),
        Some("list") => format!("usage: {PROG} list [-h]"),
        _ => USAGE.to_string(),
    };
    let prog = sub.map_or(PROG.to_string(), |s| format!("{PROG} {s}"));
    let _ = writeln!(std::io::stderr(), "{usage}\n{prog}: error: {msg}");
    2
}

fn parse(argv: &[String]) -> Result<Cmd, i32> {
    let Some(first) = argv.first() else {
        return Err(usage_error(
            None,
            "the following arguments are required: cmd",
        ));
    };
    match first.as_str() {
        "-h" | "--help" => Ok(Cmd::Help(HELP)),
        "init-key" => {
            let (mut from, mut key_file) = (None, None);
            let mut i = 1;
            while i < argv.len() {
                let a = argv[i].as_str();
                let (name, inline) = a
                    .split_once('=')
                    .map_or((a, None), |(n, v)| (n, Some(v.to_string())));
                match name {
                    "-h" | "--help" => {
                        return Ok(Cmd::Help(HELP_INIT_KEY));
                    }
                    "--from" | "--key-file" => {
                        let v = match inline {
                            Some(v) => v,
                            None => {
                                i += 1;
                                match argv.get(i) {
                                    Some(v) => v.clone(),
                                    None => {
                                        let meta = if name == "--from" {
                                            "ADDRS"
                                        } else {
                                            "KEY_FILE"
                                        };
                                        return Err(usage_error(
                                            Some("init-key"),
                                            &format!(
                                                "argument {name}: expected one argument ({meta})"
                                            ),
                                        ));
                                    }
                                }
                            }
                        };
                        if name == "--from" {
                            from = Some(v);
                        } else {
                            key_file = Some(PathBuf::from(v));
                        }
                    }
                    other => {
                        return Err(usage_error(
                            None,
                            &format!("unrecognized arguments: {other}"),
                        ));
                    }
                }
                i += 1;
            }
            match from {
                Some(from) => Ok(Cmd::InitKey { from, key_file }),
                None => Err(usage_error(
                    Some("init-key"),
                    "the following arguments are required: --from",
                )),
            }
        }
        "check" => {
            let mut timeout = CHECK_TIMEOUT_S;
            let mut i = 1;
            while i < argv.len() {
                let a = argv[i].as_str();
                let (name, inline) = a
                    .split_once('=')
                    .map_or((a, None), |(n, v)| (n, Some(v.to_string())));
                match name {
                    "-h" | "--help" => {
                        return Ok(Cmd::Help(HELP_CHECK));
                    }
                    "--timeout" => {
                        let v = match inline {
                            Some(v) => v,
                            None => {
                                i += 1;
                                argv.get(i).cloned().ok_or_else(|| {
                                    usage_error(
                                        Some("check"),
                                        "argument --timeout: expected one argument",
                                    )
                                })?
                            }
                        };
                        timeout = v.trim().parse::<f64>().map_err(|_| {
                            usage_error(
                                Some("check"),
                                &format!("argument --timeout: invalid float value: '{v}'"),
                            )
                        })?;
                    }
                    other => {
                        return Err(usage_error(
                            None,
                            &format!("unrecognized arguments: {other}"),
                        ));
                    }
                }
                i += 1;
            }
            Ok(Cmd::Check { timeout })
        }
        "list" => match argv.get(1).map(String::as_str) {
            None => Ok(Cmd::List),
            Some("-h" | "--help") => Ok(Cmd::Help(HELP_LIST)),
            Some(other) => Err(usage_error(
                None,
                &format!("unrecognized arguments: {other}"),
            )),
        },
        other => Err(usage_error(
            None,
            &format!(
                "argument cmd: invalid choice: '{other}' (choose from 'init-key', 'check', 'list')"
            ),
        )),
    }
}

fn how(ssh: Option<&str>, port: Option<u16>) -> String {
    match ssh {
        None => "this machine".into(),
        Some(dest) => format!(
            "ssh {dest}{}",
            port.map(|p| format!(" -p {p}")).unwrap_or_default()
        ),
    }
}

/// Left-justified columns separated by two spaces, trailing space stripped (cli.py `_table`).
pub fn table(rows: &[Vec<String>]) -> String {
    let n = rows.first().map_or(0, Vec::len);
    let widths: Vec<usize> = (0..n)
        .map(|i| rows.iter().map(|r| r[i].chars().count()).max().unwrap_or(0))
        .collect();
    let mut out = String::new();
    for r in rows {
        let line: Vec<String> = r
            .iter()
            .zip(&widths)
            .map(|(c, w)| format!("{c:<w$}"))
            .collect();
        out.push_str(line.join("  ").trim_end());
        out.push('\n');
    }
    out
}

fn cmd_init_key(
    cfg: &Settings,
    from: &str,
    key_file: Option<PathBuf>,
    out: &mut dyn Write,
) -> Result<i32, HostsError> {
    let key = key_file.unwrap_or_else(|| cfg.hub_key_file.clone());
    let existed = key.exists();
    let line = init_key(&key, Some(from), &cfg.remote_node_command)?;
    let k = key.display();
    let mut s = if existed {
        format!("using the existing key {k}\n")
    } else {
        format!("created {k} (private half 0600, never printed)\n")
    };
    s.push_str("\nAdd this ONE line to authorized_keys of the login user on every machine the console should manage\n");
    s.push_str("(the user that machine's node was installed for):\n\n");
    s.push_str(&format!("{line}\n"));
    if from.trim() == "*" || from.trim() == "*.*.*.*" {
        s.push_str("\nWARNING: with --from '*' a copy of this key works from anywhere (still only to start a node). Prefer this machine's address.\n");
    }
    s.push_str(&format!(
        "\nThen list the machines in {} and run: engine-console hosts check\n",
        cfg.hosts_file.display()
    ));
    let _ = out.write_all(s.as_bytes());
    Ok(0)
}

fn cmd_list(cfg: &Settings, out: &mut dyn Write) -> Result<i32, HostsError> {
    let hosts = load_hosts(&cfg.hosts_file)?;
    let note = if cfg.hosts_file.exists() {
        ""
    } else {
        " (not present: only this machine is managed)"
    };
    let mut s = format!("{}{note}\n", cfg.hosts_file.display());
    let mut rows = vec![vec![
        "ID".to_string(),
        "LABEL".into(),
        "REACHED BY".into(),
        "ENABLED".into(),
    ]];
    for h in hosts {
        let by = how(h.ssh.as_deref(), h.port) + if h.ssh_config { " (ssh_config)" } else { "" };
        rows.push(vec![
            h.id,
            h.label,
            by,
            if h.enabled { "yes" } else { "no" }.into(),
        ]);
    }
    s.push_str(&table(&rows));
    let _ = out.write_all(s.as_bytes());
    Ok(0)
}

async fn cmd_check(
    cfg: Arc<Settings>,
    timeout_s: f64,
    out: &mut dyn Write,
) -> Result<i32, HostsError> {
    let fleet = FleetService::new(Arc::clone(&cfg), Arc::new(EventBus::new()))?;
    fleet.start().await;
    let hosts = crate::domain::fleet::FleetPort::hosts(&fleet);
    let mut snaps: Vec<(String, HostSummary)> = Vec::new();
    let deadline = Instant::now() + Duration::from_secs_f64(timeout_s.max(0.0));
    while Instant::now() < deadline && snaps.len() < hosts.len() {
        for s in fleet.summaries().unwrap_or_default() {
            // remember a host the moment its first attempt has an outcome (a failed link flips back to connecting)
            if !snaps.iter().any(|(id, _)| *id == s.id) && SETTLED.contains(&s.link.state.as_str())
            {
                snaps.push((s.id.clone(), s));
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let last = fleet.summaries();
    fleet.stop().await;
    let last = match last {
        Ok(l) => l,
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "engine-console: {e}");
            return Ok(1);
        }
    };
    for s in last {
        if !snaps.iter().any(|(id, _)| *id == s.id) {
            snaps.push((s.id.clone(), s));
        }
    }
    let mut rows = vec![
        ["HOST", "STATE", "NODE", "ARCH", "GPUS", "POLICY"]
            .map(String::from)
            .to_vec(),
    ];
    let mut detail = Vec::new();
    let mut bad = 0;
    for h in &hosts {
        let Some((_, s)) = snaps.iter().find(|(id, _)| *id == h.id) else {
            continue;
        };
        let up = s.link.state == "up" || s.link.state == "stale";
        if !up && s.link.state != "disabled" {
            bad += 1;
        }
        let node = s.node.as_ref();
        rows.push(vec![
            h.id.clone(),
            s.link.state.clone(),
            node.and_then(|n| n.version.clone())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "-".into()),
            node.and_then(|n| n.arch.clone())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "-".into()),
            if up {
                s.gpus.len().to_string()
            } else {
                "-".into()
            },
            s.policy
                .as_ref()
                .map(|p| p.mode.clone())
                .filter(|m| !m.is_empty())
                .unwrap_or_else(|| "-".into()),
        ]);
        if let Some(e) = &s.link.error {
            detail.push(format!("  {}: {e}", h.id));
        } else if !up && s.link.state == "connecting" {
            detail.push(format!("  {}: no answer yet", h.id));
        }
        for n in &s.link.notes {
            detail.push(format!("  {}: note: {n}", h.id));
        }
        if let Some(e) = s.policy.as_ref().and_then(|p| p.error.clone()) {
            detail.push(format!("  {}: node policy: {e}", h.id));
        }
    }
    let mut text = table(&rows);
    for d in detail {
        text.push_str(&d);
        text.push('\n');
    }
    let _ = out.write_all(text.as_bytes());
    Ok(i32::from(bad > 0))
}

/// Python's logging "last resort": the `hosts` commands configure no logging, so a record of WARNING or worse from
/// any logger (the fleet link's `link <host>: down: ... (retry in N s)` during `hosts check`) is written to stderr as
/// its bare message, one line each, and everything below WARNING is dropped.
struct LastResort;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for LastResort {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if *event.metadata().level() > tracing::Level::WARN {
            return; // tracing orders by verbosity: INFO, DEBUG and TRACE are "greater" than WARN
        }
        let mut m = MessageOnly(String::new());
        event.record(&mut m);
        let _ = writeln!(std::io::stderr(), "{}", m.0);
    }
}

/// The formatted `message` of an event (`%(message)s`), without its other fields.
struct MessageOnly(String);

impl tracing::field::Visit for MessageOnly {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            value.clone_into(&mut self.0);
        }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }
}

/// Install the last-resort handler for this process (the CLI's only subscriber; a second call is a no-op).
fn install_last_resort() {
    use tracing_subscriber::layer::SubscriberExt;
    let _ =
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(LastResort));
}

/// Run one `engine-console hosts` command; the return value is the exit code.
pub fn hosts_cli(argv: &[String]) -> i32 {
    install_last_resort();
    let cmd = match parse(argv) {
        Ok(c) => c,
        Err(code) => return code,
    };
    if let Cmd::Help(text) = cmd {
        let _ = std::io::stdout().write_all(text.as_bytes());
        return 0;
    }
    let cfg = match Settings::from_env() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            let _ = writeln!(
                std::io::stderr(),
                "engine-console: invalid setting {}: {}",
                e.field,
                e.message
            );
            return 1;
        }
    };
    hosts_cli_with(&cmd, &cfg, &mut std::io::stdout())
}

fn hosts_cli_with(cmd: &Cmd, cfg: &Arc<Settings>, out: &mut dyn Write) -> i32 {
    let res = match cmd {
        Cmd::InitKey { from, key_file } => cmd_init_key(cfg, from, key_file.clone(), out),
        Cmd::List => cmd_list(cfg, out),
        Cmd::Check { timeout } => match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt.block_on(cmd_check(Arc::clone(cfg), *timeout, out)),
            Err(_) => Ok(1),
        },
        Cmd::Help(_) => Ok(0),
    };
    match res {
        Ok(code) => code,
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "engine-console: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn test_parse_usage_errors_exit_2() {
        assert_eq!(parse(&args(&[])), Err(2));
        assert_eq!(parse(&args(&["bogus"])), Err(2));
        assert_eq!(parse(&args(&["init-key"])), Err(2));
        assert_eq!(parse(&args(&["check", "--timeout", "x"])), Err(2));
        assert_eq!(parse(&args(&["list"])), Ok(Cmd::List));
        assert_eq!(
            parse(&args(&["check", "--timeout=2.5"])),
            Ok(Cmd::Check { timeout: 2.5 })
        );
        assert_eq!(
            parse(&args(&["init-key", "--from", "10.0.0.5"])),
            Ok(Cmd::InitKey {
                from: "10.0.0.5".into(),
                key_file: None
            })
        );
    }

    #[test]
    fn test_list_without_hosts_file() {
        let d = tempfile::tempdir().unwrap();
        let mut cfg = Settings::defaults();
        cfg.hosts_file = d.path().join("hosts.json");
        let mut out = Vec::new();
        assert_eq!(hosts_cli_with(&Cmd::List, &Arc::new(cfg), &mut out), 0);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("(not present: only this machine is managed)"));
        assert!(text.contains("ID     LABEL"));
        assert!(text.contains("local  "));
        assert!(text.contains("this machine  yes"));
    }

    #[test]
    fn test_table_alignment() {
        let t = table(&[
            vec!["A".into(), "BB".into()],
            vec!["ccc".into(), "d".into()],
        ]);
        assert_eq!(t, "A    BB\nccc  d\n");
    }
}
