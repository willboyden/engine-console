//! Port of node/src/ecnode/__main__.py: the `ecnode` command line, argparse-compatible.
//!
//!   ecnode --stdio [--policy PATH]     speak the hub protocol on stdin/stdout (what the hub starts)
//!   ecnode check [--policy PATH]       print what this machine would report (read-only)
//!   ecnode policy-template             print a starter monitor-mode policy (writes nothing)
//!   ecnode version
//!
//! Help and usage-error texts are Python 3.12 argparse's at 80 columns; exit codes are argparse's (0 for --help,
//! 2 for a usage error).

use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use ec_proto::Obj;
use serde_json::json;

use crate::agent::{Agent, VERSION};
use crate::policy::PolicyStore;
use crate::server::{Server, ServerOptions};

pub const USAGE: &str = "usage: ecnode [-h] [--stdio] [--policy PATH]\n              {check,policy-template,version} ...\n";
pub const HELP: &str = "usage: ecnode [-h] [--stdio] [--policy PATH]
              {check,policy-template,version} ...

Engine Console node agent (one per machine).

positional arguments:
  {check,policy-template,version}
    check               print what this machine would report to a hub (read-
                        only)
    policy-template     print a starter monitor-mode policy (writes nothing)
    version             print the version

options:
  -h, --help            show this help message and exit
  --stdio               speak the hub protocol on stdin/stdout
  --policy PATH         policy file (default: /etc/ecnode/policy.json, then
                        ~/.config/ecnode/policy.json)
";
const CHECK_USAGE: &str = "usage: ecnode check [-h] [--policy PATH]\n";
const CHECK_HELP: &str = "usage: ecnode check [-h] [--policy PATH]

options:
  -h, --help     show this help message and exit
  --policy PATH
";
const PLAIN_HELP: &str = "usage: ecnode {cmd} [-h]

options:
  -h, --help  show this help message and exit
";

/// What the command line asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Stdio {
        policy: Option<String>,
    },
    Check {
        policy: Option<String>,
    },
    PolicyTemplate,
    Version,
    /// No command: help on stderr, exit 2.
    Nothing,
}

/// A parse outcome that ends the program before running anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exit {
    /// text for stdout, exit 0
    Help(String),
    /// text for stderr, exit 2
    Usage(String),
}

fn usage_error(usage: &str, prog: &str, msg: &str) -> Exit {
    Exit::Usage(format!("{usage}{prog}: error: {msg}\n"))
}

/// Resolve a long option the way argparse does: an exact name, or a unique prefix of one.
fn long_opt<'a>(arg: &str, names: &[&'a str]) -> Option<&'a str> {
    if let Some(n) = names.iter().find(|n| **n == arg) {
        return Some(n);
    }
    let hits: Vec<&&str> = names.iter().filter(|n| n.starts_with(arg)).collect();
    if arg.len() > 2 && hits.len() == 1 {
        Some(hits[0])
    } else {
        None
    }
}

/// argparse for `ecnode`.
pub fn parse(args: &[String]) -> Result<Command, Exit> {
    let mut stdio = false;
    let mut policy: Option<String> = None;
    let mut cmd: Option<String> = None;
    let mut sub_policy: Option<String> = None;
    let mut unknown: Vec<String> = Vec::new();
    let mut i = 0;
    // top level, until the sub-command
    while i < args.len() {
        let a = &args[i];
        let (name, inline) = match a.split_once('=') {
            Some((n, v)) if a.starts_with("--") => (n, Some(v.to_string())),
            _ => (a.as_str(), None),
        };
        if a == "-h"
            || (a.starts_with("--") && long_opt(name, &["--help"]).is_some() && inline.is_none())
        {
            return Err(Exit::Help(HELP.into()));
        }
        if a.starts_with("--") {
            match long_opt(name, &["--help", "--stdio", "--policy"]) {
                Some("--stdio") if inline.is_none() => stdio = true,
                Some("--policy") => {
                    if let Some(v) = inline {
                        policy = Some(v);
                    } else if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        i += 1;
                        policy = Some(args[i].clone());
                    } else {
                        return Err(usage_error(
                            USAGE,
                            "ecnode",
                            "argument --policy: expected one argument",
                        ));
                    }
                }
                _ => unknown.push(a.clone()),
            }
            i += 1;
            continue;
        }
        if a.starts_with('-') && a.len() > 1 {
            unknown.push(a.clone());
            i += 1;
            continue;
        }
        cmd = Some(a.clone());
        i += 1;
        break;
    }
    if let Some(c) = &cmd {
        if !["check", "policy-template", "version"].contains(&c.as_str()) {
            return Err(usage_error(
                USAGE,
                "ecnode",
                &format!(
                    "argument cmd: invalid choice: {} (choose from check, policy-template, version)",
                    ec_proto::pyfmt::repr_str(c)
                ),
            ));
        }
        // the sub-command's own arguments
        while i < args.len() {
            let a = &args[i];
            let (name, inline) = match a.split_once('=') {
                Some((n, v)) if a.starts_with("--") => (n, Some(v.to_string())),
                _ => (a.as_str(), None),
            };
            let names: &[&str] = if c == "check" {
                &["--help", "--policy"]
            } else {
                &["--help"]
            };
            let hit = if a.starts_with("--") {
                long_opt(name, names)
            } else {
                None
            };
            if a == "-h" || (hit == Some("--help") && inline.is_none()) {
                return Err(Exit::Help(if c == "check" {
                    CHECK_HELP.into()
                } else {
                    PLAIN_HELP.replace("{cmd}", c)
                }));
            }
            if hit == Some("--policy") {
                if let Some(v) = inline {
                    sub_policy = Some(v);
                } else if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    i += 1;
                    sub_policy = Some(args[i].clone());
                } else {
                    return Err(usage_error(
                        CHECK_USAGE,
                        "ecnode check",
                        "argument --policy: expected one argument",
                    ));
                }
            } else {
                unknown.push(a.clone());
            }
            i += 1;
        }
    }
    if !unknown.is_empty() {
        return Err(usage_error(
            USAGE,
            "ecnode",
            &format!("unrecognized arguments: {}", unknown.join(" ")),
        ));
    }
    if stdio {
        if cmd.is_some() {
            return Err(usage_error(USAGE, "ecnode", "--stdio takes no command"));
        }
        return Ok(Command::Stdio { policy });
    }
    Ok(match cmd.as_deref() {
        Some("version") => Command::Version,
        Some("check") => Command::Check {
            policy: sub_policy.or(policy),
        },
        Some("policy-template") => Command::PolicyTemplate,
        _ => Command::Nothing,
    })
}

/// Run the command line; returns the exit code. `out`/`err` are stdout/stderr (tests capture them).
pub fn main_with(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    match parse(args) {
        Err(Exit::Help(text)) => {
            let _ = out.write_all(text.as_bytes());
            0
        }
        Err(Exit::Usage(text)) => {
            let _ = err.write_all(text.as_bytes());
            2
        }
        Ok(Command::Version) => {
            let _ = writeln!(out, "ecnode {VERSION}");
            0
        }
        Ok(Command::Nothing) => {
            let _ = err.write_all(HELP.as_bytes());
            2
        }
        Ok(Command::Check { policy }) => crate::cli::check(policy.as_deref(), out),
        Ok(Command::PolicyTemplate) => crate::cli::policy_template(out),
        Ok(Command::Stdio { policy }) => stdio(policy),
    }
}

/// `--stdio`: the protocol owns the original stdin/stdout. Everything else (a stray print, a child process that
/// forgets to redirect) gets /dev/null as stdin and stderr as stdout, so it can never corrupt the frame stream.
fn stdio(policy: Option<String>) -> i32 {
    let (proto_in, proto_out) = match (
        nix::unistd::dup(std::io::stdin()),
        nix::unistd::dup(std::io::stdout()),
    ) {
        (Ok(i), Ok(o)) => (File::from(i), File::from(o)),
        _ => {
            crate::server::say("cannot duplicate stdin/stdout");
            return 2;
        }
    };
    let redirected = File::open("/dev/null")
        .map_err(|_| ())
        .and_then(|devnull| nix::unistd::dup2_stdin(&devnull).map_err(|_| ()))
        .and_then(|()| nix::unistd::dup2_stdout(std::io::stderr()).map_err(|_| ()));
    if redirected.is_err() {
        crate::server::say("cannot protect the protocol stream (dup2 failed)");
        return 2;
    }
    let agent = std::panic::catch_unwind(|| Agent::new(PolicyStore::new(policy)));
    let Ok(agent) = agent else {
        let mut fatal = Obj::new();
        fatal.insert("type".into(), json!("fatal"));
        fatal.insert("error".into(), json!("the node could not start: panic"));
        let mut proto_out = proto_out;
        let _ = proto_out.write_all(ec_proto::MAGIC);
        if let Ok(f) = ec_proto::encode_json(&fatal) {
            let _ = proto_out.write_all(&f);
        }
        let _ = proto_out.flush();
        return 2;
    };
    Server::new(Arc::new(agent), ServerOptions::default())
        .run(Box::new(proto_in), Box::new(proto_out))
}
