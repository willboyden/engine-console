//! Port of backend/tests/fleet/test_hosts.py: hosts.json loading, the exact ssh command line, the scrubbed
//! environments, and the hub's ssh key.
//!
//! Python patches `socket.gethostname`, `os.geteuid` and the process environment; the Rust tests use the seams
//! `default_hosts_named`, `load_hosts_as`, `ssh_env_from` and `local_env_from` (the public functions call them with
//! the real values). ssh-keygen runs for real, in a temp dir; `ssh -G` evaluates a command line without connecting.

#![allow(
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::double_ended_iterator_last,
    clippy::assertions_on_constants,
    clippy::float_cmp
)]

use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ec_hub::config::Settings;
use ec_hub::domain::fleet::HostConfig;
use ec_hub::services::hosts::{
    DEFAULT_REMOTE_COMMAND, HostsError, LOCAL_ENV_KEYS, default_hosts, default_hosts_named,
    init_key, init_key_with, key_line, load_hosts, load_hosts_as, local_argv, local_env_from,
    parse_hosts, py_isprintable, sibling_ecnode, ssh_argv, ssh_env, ssh_env_from,
};
use serde_json::{Value, json};

const REMOTE_CMD: &str = "/usr/local/lib/ecnode/ecnode --stdio";
/// A well-formed ed25519 public key whose 32 bytes spell "FAKE-TEST-KEY-NOT-A-REAL-ONE-000": obviously not a real key.
const PUB: &str = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEZBS0UtVEVTVC1LRVktTk9ULUEtUkVBTC1PTkUtMDAw user@somewhere";

/// init_key changes the process umask while ssh-keygen runs: the tests that create keys take turns.
static UMASK: Mutex<()> = Mutex::new(());

fn write_hosts(path: &Path, doc: &Value, mode: u32) -> PathBuf {
    let text = match doc {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    std::fs::write(path, text).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    path.to_path_buf()
}

fn host(kw: Value) -> Value {
    let mut h = json!({"id": "gpu-box-2", "ssh": "ops@10.0.0.5"});
    for (k, v) in kw.as_object().unwrap() {
        h[k] = v.clone();
    }
    h
}

fn cfg(
    id: &str,
    label: &str,
    ssh: Option<&str>,
    port: Option<u16>,
    ssh_config: bool,
    enabled: bool,
) -> HostConfig {
    HostConfig {
        id: id.into(),
        label: label.into(),
        ssh: ssh.map(str::to_string),
        port,
        ssh_config,
        enabled,
    }
}

fn err_text(e: &HostsError) -> String {
    e.to_string()
}

// ── hosts.json: what is accepted ───────────────────────────────────────────────────────────────────────
#[test]
fn test_no_file_means_this_machine_only() {
    assert_eq!(
        default_hosts_named("my-workstation"),
        vec![cfg("local", "my-workstation", None, None, false, true)]
    );
    let d = tempfile::tempdir().unwrap();
    assert_eq!(
        load_hosts(&d.path().join("absent.json")).unwrap(),
        default_hosts()
    );
}

#[test]
fn test_no_file_label_is_never_empty_or_unprintable() {
    assert_eq!(default_hosts_named("")[0].label, "This machine");
    let label = default_hosts_named(&format!("ok\u{7}name\n{}", "x".repeat(100)))[0]
        .label
        .clone();
    assert!(label.chars().all(py_isprintable));
    assert!((1..=64).contains(&label.chars().count()));
}

#[test]
fn test_a_full_file() {
    let d = tempfile::tempdir().unwrap();
    let doc = json!({"hosts": [
        {"id": "local", "label": "Workstation"},
        {"id": "gpu-box-2", "label": "GPU box 2", "ssh": "ops@10.0.0.5", "port": 2222},
        {"id": "edge", "ssh": "edge-alias", "ssh_config": true, "enabled": false},
    ]});
    let got = load_hosts(&write_hosts(&d.path().join("hosts.json"), &doc, 0o600)).unwrap();
    assert_eq!(
        got,
        vec![
            cfg("local", "Workstation", None, None, false, true),
            cfg(
                "gpu-box-2",
                "GPU box 2",
                Some("ops@10.0.0.5"),
                Some(2222),
                false,
                true
            ),
            cfg("edge", "edge", Some("edge-alias"), None, true, false), // label defaults to the id
        ]
    );
}

#[test]
fn test_leaving_local_out_is_allowed() {
    let d = tempfile::tempdir().unwrap();
    let got = load_hosts(&write_hosts(
        &d.path().join("h.json"),
        &json!({"hosts": [host(json!({}))]}),
        0o600,
    ))
    .unwrap();
    assert_eq!(
        got.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        ["gpu-box-2"]
    );
}

#[test]
fn test_modes_that_nobody_else_can_write_are_fine() {
    for mode in [0o600, 0o640, 0o400, 0o644] {
        let d = tempfile::tempdir().unwrap();
        assert!(
            !load_hosts(&write_hosts(
                &d.path().join("h.json"),
                &json!({"hosts": [host(json!({}))]}),
                mode
            ))
            .unwrap()
            .is_empty()
        );
    }
}

#[test]
fn test_32_hosts_is_the_limit() {
    let d = tempfile::tempdir().unwrap();
    let ok: Vec<Value> = (0..32)
        .map(|i| host(json!({"id": format!("h{i}")})))
        .collect();
    assert_eq!(
        load_hosts(&write_hosts(
            &d.path().join("ok.json"),
            &json!({"hosts": ok}),
            0o600
        ))
        .unwrap()
        .len(),
        32
    );
    let many: Vec<Value> = (0..33)
        .map(|i| host(json!({"id": format!("h{i}")})))
        .collect();
    let e = load_hosts(&write_hosts(
        &d.path().join("many.json"),
        &json!({"hosts": many}),
        0o600,
    ))
    .unwrap_err();
    assert!(err_text(&e).contains("1-32"));
}

// ── hosts.json: what is refused ────────────────────────────────────────────────────────────────────────
#[test]
fn test_refused_documents() {
    let cases: Vec<(Value, &str)> = vec![
        (json!("{not json"), "not valid JSON"),
        (json!(""), "not valid JSON"),
        (json!("[]"), "nothing else"),
        (json!("\"hosts\""), "nothing else"),
        (
            json!({"hosts": [host(json!({}))], "extra": 1}),
            "nothing else",
        ),
        (json!({"hosts": {}}), "1-32"),
        (json!({"hosts": []}), "1-32"),
        (json!({"hosts": ["local"]}), "must be an object"),
        (
            json!({"hosts": [host(json!({})), host(json!({}))]}),
            "used twice",
        ),
        (
            json!({"hosts": [host(json!({"colour": "red"}))]}),
            "unknown key",
        ),
        (json!({"hosts": [{"ssh": "ops@10.0.0.5"}]}), "id must be"),
        (
            json!({"hosts": [host(json!({"id": "Upper"}))]}),
            "id must be",
        ),
        (
            json!({"hosts": [host(json!({"id": "-lead"}))]}),
            "id must be",
        ),
        (
            json!({"hosts": [host(json!({"id": "a".repeat(33)}))]}),
            "id must be",
        ),
        (json!({"hosts": [host(json!({"id": "a\n"}))]}), "id must be"), // a whole-string match: no trailing newline
        (json!({"hosts": [host(json!({"label": ""}))]}), "label"),
        (
            json!({"hosts": [host(json!({"label": "x".repeat(65)}))]}),
            "label",
        ),
        (
            json!({"hosts": [host(json!({"label": "bad\u{7}bell"}))]}),
            "label",
        ),
        (json!({"hosts": [host(json!({"label": 7}))]}), "label"),
        (
            json!({"hosts": [host(json!({"enabled": "yes"}))]}),
            "true or false",
        ),
        (
            json!({"hosts": [host(json!({"ssh_config": 1}))]}),
            "true or false",
        ),
        (
            json!({"hosts": [{"id": "local", "ssh": "ops@10.0.0.5"}]}),
            "takes no ssh",
        ),
        (
            json!({"hosts": [{"id": "local", "port": 22}]}),
            "takes no ssh",
        ),
        (
            json!({"hosts": [{"id": "local", "ssh_config": true}]}),
            "takes no ssh",
        ),
        (json!({"hosts": [{"id": "gpu-box-2"}]}), "ssh must be"),
        (
            json!({"hosts": [host(json!({"ssh": null}))]}),
            "ssh must be",
        ),
        (json!({"hosts": [host(json!({"ssh": 5}))]}), "ssh must be"),
        (
            json!({"hosts": [host(json!({"ssh": "-oProxyCommand=touch${IFS}x"}))]}),
            "ssh must be",
        ),
        (
            json!({"hosts": [host(json!({"ssh": "ops@-host"}))]}),
            "ssh must be",
        ),
        (
            json!({"hosts": [host(json!({"ssh": "ops@host%h"}))]}),
            "ssh must be",
        ),
        (
            json!({"hosts": [host(json!({"ssh": "ops@host name"}))]}),
            "ssh must be",
        ),
        (
            json!({"hosts": [host(json!({"ssh": "ops@host\n"}))]}),
            "ssh must be",
        ), // a trailing newline must not slip through
        (
            json!({"hosts": [host(json!({"ssh": "ops@host;reboot"}))]}),
            "ssh must be",
        ),
        (
            json!({"hosts": [host(json!({"ssh": format!("{}@host", "a".repeat(33))}))]}),
            "ssh must be",
        ),
        (
            json!({"hosts": [host(json!({"ssh": "h".repeat(254)}))]}),
            "ssh must be",
        ),
        (json!({"hosts": [host(json!({"port": 0}))]}), "port"),
        (json!({"hosts": [host(json!({"port": 65536}))]}), "port"),
        (json!({"hosts": [host(json!({"port": true}))]}), "port"),
        (json!({"hosts": [host(json!({"port": "22"}))]}), "port"),
        (json!({"hosts": [host(json!({"port": 22.5}))]}), "port"),
    ];
    for (doc, needle) in cases {
        let d = tempfile::tempdir().unwrap();
        let e = load_hosts(&write_hosts(&d.path().join("hosts.json"), &doc, 0o600)).unwrap_err();
        assert!(err_text(&e).contains(needle), "{doc}: {e}");
    }
}

#[test]
fn test_non_finite_numbers_are_rejected() {
    for constant in ["NaN", "Infinity", "-Infinity", "1e999"] {
        let d = tempfile::tempdir().unwrap();
        let text = format!(
            "{{\"hosts\": [{{\"id\": \"gpu-box-2\", \"ssh\": \"ops@10.0.0.5\", \"port\": {constant}}}]}}"
        );
        let e = load_hosts(&write_hosts(
            &d.path().join("hosts.json"),
            &json!(text),
            0o600,
        ))
        .unwrap_err();
        let msg = err_text(&e);
        // the decoder hook's own words (Python: parse_constant / parse_float)
        assert!(
            msg.contains("is not allowed") || msg.contains("is not a finite number"),
            "{msg}"
        );
    }
}

#[test]
fn test_duplicate_keys_in_one_object_are_rejected() {
    let d = tempfile::tempdir().unwrap();
    let text = r#"{"hosts": [{"id": "a", "id": "b", "ssh": "ops@10.0.0.5"}]}"#;
    let e = load_hosts(&write_hosts(
        &d.path().join("hosts.json"),
        &json!(text),
        0o600,
    ))
    .unwrap_err();
    assert!(err_text(&e).contains("appears twice"), "{e}");
    assert!(
        err_text(&e).ends_with(": the key 'id' appears twice in one object"),
        "{e}"
    );
}

#[test]
fn test_absurd_nesting_is_an_error_not_a_crash() {
    let d = tempfile::tempdir().unwrap();
    let e = load_hosts(&write_hosts(
        &d.path().join("hosts.json"),
        &json!("[".repeat(60_000)),
        0o600,
    ))
    .unwrap_err();
    assert!(err_text(&e).contains("not valid JSON"));
}

#[test]
fn test_huge_integers_are_an_error() {
    let d = tempfile::tempdir().unwrap();
    let text = format!(
        "{{\"hosts\": [{{\"id\": \"a\", \"ssh\": \"x\", \"port\": {}}}]}}",
        "9".repeat(6000)
    );
    assert!(
        load_hosts(&write_hosts(
            &d.path().join("hosts.json"),
            &json!(text),
            0o600
        ))
        .is_err()
    );
}

#[test]
fn test_invalid_utf8_is_an_error() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("hosts.json");
    std::fs::write(&p, b"{\"hosts\": [{\"id\": \"a\xff\"}]}").unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert!(err_text(&load_hosts(&p).unwrap_err()).contains("not valid JSON"));
}

#[test]
fn test_error_messages_name_the_file() {
    let d = tempfile::tempdir().unwrap();
    let p = write_hosts(&d.path().join("hosts.json"), &json!({"hosts": []}), 0o600);
    assert!(err_text(&load_hosts(&p).unwrap_err()).starts_with(&p.display().to_string()));
}

// ── hosts.json: the file itself must be trustworthy ────────────────────────────────────────────────────
#[test]
fn test_group_or_world_writable_is_refused() {
    for mode in [0o620, 0o602, 0o666, 0o664] {
        let d = tempfile::tempdir().unwrap();
        let p = write_hosts(
            &d.path().join("hosts.json"),
            &json!({"hosts": [host(json!({}))]}),
            mode,
        );
        assert!(err_text(&load_hosts(&p).unwrap_err()).contains("group- or world-writable"));
    }
}

#[test]
fn test_a_symlink_is_refused_even_to_a_good_file() {
    let d = tempfile::tempdir().unwrap();
    let real = write_hosts(
        &d.path().join("real.json"),
        &json!({"hosts": [host(json!({}))]}),
        0o600,
    );
    let link = d.path().join("hosts.json");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    assert!(err_text(&load_hosts(&link).unwrap_err()).contains("symlink"));
}

#[test]
fn test_a_dangling_symlink_is_refused_not_treated_as_absent() {
    let d = tempfile::tempdir().unwrap();
    let link = d.path().join("hosts.json");
    std::os::unix::fs::symlink(d.path().join("nowhere.json"), &link).unwrap();
    assert!(err_text(&load_hosts(&link).unwrap_err()).contains("symlink"));
}

#[test]
fn test_a_file_owned_by_someone_else_is_refused() {
    use std::os::unix::fs::MetadataExt as _;
    let d = tempfile::tempdir().unwrap();
    let p = write_hosts(
        &d.path().join("hosts.json"),
        &json!({"hosts": [host(json!({}))]}),
        0o600,
    );
    let uid = std::fs::metadata(&p).unwrap().uid();
    assert!(err_text(&load_hosts_as(&p, uid + 1).unwrap_err()).contains("owned by"));
    assert!(load_hosts_as(&p, uid).is_ok());
}

#[test]
fn test_a_directory_and_a_fifo_are_refused_without_hanging() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("dir.json");
    std::fs::create_dir(&dir).unwrap();
    assert!(err_text(&load_hosts(&dir).unwrap_err()).contains("regular file"));
    let fifo = d.path().join("fifo.json");
    nix::unistd::mkfifo(&fifo, nix::sys::stat::Mode::from_bits_truncate(0o600)).unwrap();
    assert!(err_text(&load_hosts(&fifo).unwrap_err()).contains("regular file"));
}

#[test]
fn test_size_cap() {
    let d = tempfile::tempdir().unwrap();
    let doc = json!({"hosts": [host(json!({}))]}).to_string();
    let p = write_hosts(
        &d.path().join("hosts.json"),
        &json!(format!("{doc}{}", " ".repeat(64 * 1024))),
        0o600,
    );
    assert!(err_text(&load_hosts(&p).unwrap_err()).contains("over 64 KiB"));
    let small = write_hosts(
        &d.path().join("ok.json"),
        &json!(format!("{doc}{}", " ".repeat(1000))),
        0o600,
    );
    assert!(!load_hosts(&small).unwrap().is_empty());
}

#[test]
fn test_an_unreadable_file_is_an_error_not_a_default() {
    let d = tempfile::tempdir().unwrap();
    let p = write_hosts(
        &d.path().join("hosts.json"),
        &json!({"hosts": [host(json!({}))]}),
        0o000,
    );
    if std::fs::File::open(&p).is_ok() {
        return; // running as root: the mode does not stop us, so there is nothing to test
    }
    assert!(err_text(&load_hosts(&p).unwrap_err()).contains("cannot open"));
}

#[test]
fn test_parse_hosts_is_usable_on_its_own() {
    assert_eq!(
        parse_hosts(&json!({"hosts": [host(json!({}))]})).unwrap()[0]
            .ssh
            .as_deref(),
        Some("ops@10.0.0.5")
    );
}

// ── the ssh command line ───────────────────────────────────────────────────────────────────────────────
const OPTIONS: &[&str] = &[
    "BatchMode=yes",
    "StrictHostKeyChecking=yes",
    "UpdateHostKeys=no",
    "VerifyHostKeyDNS=no",
    "IdentityAgent=none",
    "PKCS11Provider=none",
    "PasswordAuthentication=no",
    "KbdInteractiveAuthentication=no",
    "ControlMaster=no",
    "ControlPath=none",
    "ForwardAgent=no",
    "ForwardX11=no",
    "ClearAllForwardings=yes",
    "PermitLocalCommand=no",
    "ConnectTimeout=5",
    "ServerAliveInterval=5",
    "ServerAliveCountMax=3",
];

fn expected_argv(key: &str, dest: &str, port: Option<u16>, with_config: bool) -> Vec<String> {
    let mut argv: Vec<String> = ["ssh", "-T", "-e", "none"].map(String::from).to_vec();
    if !with_config {
        argv.extend(["-F".into(), "none".into()]);
    }
    for o in OPTIONS {
        argv.extend(["-o".into(), (*o).to_string()]);
    }
    argv.extend([
        "-o".into(),
        format!("IdentityFile={key}"),
        "-o".into(),
        "IdentitiesOnly=yes".into(),
    ]);
    if let Some(p) = port {
        argv.extend(["-p".into(), p.to_string()]);
    }
    argv.extend(["--".into(), dest.into(), REMOTE_CMD.into()]);
    argv
}

/// A stand-in key file: ssh_argv only needs it to exist (ssh reads it later), and it is never a real key.
fn hub_key(dir: &Path) -> PathBuf {
    let key = dir.join("hub-key");
    std::fs::write(&key, "not a real key\n").unwrap();
    key
}

#[test]
fn test_a_key_path_ssh_would_parse_as_syntax_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let h = cfg(
        "gpu-box-2",
        "GPU box 2",
        Some("me@10.0.0.5"),
        None,
        false,
        true,
    );
    for name in ["with space", "quote'd", "pct%d"] {
        let key = d.path().join(name);
        std::fs::write(&key, "x").unwrap();
        assert!(ssh_argv(&h, &key, REMOTE_CMD).is_err(), "{name}");
    }
}

#[test]
fn test_ssh_argv_is_exactly_the_documented_command() {
    let d = tempfile::tempdir().unwrap();
    let key = hub_key(d.path());
    let h = cfg(
        "gpu-box-2",
        "GPU box 2",
        Some("ops@10.0.0.5"),
        None,
        false,
        true,
    );
    assert_eq!(
        ssh_argv(&h, &key, REMOTE_CMD).unwrap(),
        expected_argv(key.to_str().unwrap(), "ops@10.0.0.5", None, false)
    );
}

#[test]
fn test_ssh_argv_port_only_when_set_and_config_only_when_asked() {
    let d = tempfile::tempdir().unwrap();
    let key = hub_key(d.path());
    let k = key.to_str().unwrap();
    let h = cfg(
        "gpu-box-2",
        "x",
        Some("ops@10.0.0.5"),
        Some(2222),
        false,
        true,
    );
    assert_eq!(
        ssh_argv(&h, &key, REMOTE_CMD).unwrap(),
        expected_argv(k, "ops@10.0.0.5", Some(2222), false)
    );
    let a = cfg("alias", "x", Some("edge-alias"), None, true, true);
    let argv = ssh_argv(&a, &key, REMOTE_CMD).unwrap();
    assert_eq!(argv, expected_argv(k, "edge-alias", None, true));
    assert!(!argv.contains(&"-F".to_string()) && !argv.contains(&"-p".to_string()));
}

#[test]
fn test_the_destination_follows_a_double_dash_and_is_never_an_option() {
    let d = tempfile::tempdir().unwrap();
    let key = hub_key(d.path());
    let argv = ssh_argv(
        &cfg("a", "a", Some("ops@10.0.0.5"), None, false, true),
        &key,
        REMOTE_CMD,
    )
    .unwrap();
    assert_eq!(argv[argv.len() - 3..], ["--", "ops@10.0.0.5", REMOTE_CMD]);
    assert_eq!(argv.iter().filter(|a| *a == "--").count(), 1);
}

#[test]
fn test_ssh_argv_refuses_a_destination_that_is_not_a_plain_name() {
    let d = tempfile::tempdir().unwrap();
    let key = hub_key(d.path());
    for bad in [
        "-oProxyCommand=evil",
        "-J jump",
        "ops@-host",
        "-ops@10.0.0.5",
        ".ops@host",
        "a b",
        "host\n",
        "",
        "x;y",
        "host%h",
    ] {
        let h = cfg("a", "a", Some(bad), None, false, true); // bypasses the loader's checks
        assert!(ssh_argv(&h, &key, REMOTE_CMD).is_err(), "{bad:?}");
    }
}

#[test]
fn test_ssh_argv_refuses_local_and_odd_key_paths_and_commands() {
    let d = tempfile::tempdir().unwrap();
    let key = hub_key(d.path());
    let e = ssh_argv(
        &cfg("local", "l", None, None, false, true),
        &key,
        REMOTE_CMD,
    )
    .unwrap_err();
    assert!(err_text(&e).contains("without ssh"));
    let h = cfg("a", "a", Some("ops@10.0.0.5"), None, false, true);
    for k in ["", "-key", "k\n", "k\u{0}"] {
        let e = ssh_argv(&h, Path::new(k), REMOTE_CMD).unwrap_err();
        assert!(err_text(&e).contains("key path"), "{k:?}: {e}");
    }
    let long = format!("/x {}", "a".repeat(100 * 100));
    for c in [
        "",
        "ecnode --stdio",
        "/x; reboot",
        "/x\"y",
        "/x --a\nb",
        "/x $(id)",
        "-/x",
        long.as_str(),
    ] {
        let e = ssh_argv(&h, &key, c).unwrap_err();
        assert!(err_text(&e).contains("remote node command"), "{c:?}: {e}");
    }
}

/// With a missing identity file the real ssh only warns and then offers the user's default identity files, so the
/// command must never be built: the console would log in with a key that is not restricted to the node.
#[test]
fn test_ssh_argv_refuses_to_build_a_command_without_the_hub_key() {
    let d = tempfile::tempdir().unwrap();
    let h = cfg("a", "a", Some("ops@10.0.0.5"), None, false, true);
    let e = ssh_argv(&h, &d.path().join("missing").join("hub-key"), REMOTE_CMD).unwrap_err();
    assert!(matches!(e, HostsError::NoHubKey(_)) && err_text(&e).contains("hosts init-key"));
    std::fs::create_dir(d.path().join("a-directory")).unwrap();
    assert!(matches!(
        ssh_argv(&h, &d.path().join("a-directory"), REMOTE_CMD).unwrap_err(),
        HostsError::NoHubKey(_)
    ));
}

/// `ssh -G` evaluates a command line without connecting, so it proves the options are ones ssh knows (a typo is "Bad
/// configuration option") and that they take effect. `-F none` keeps it from reading any user or system config.
#[test]
fn test_the_real_ssh_client_accepts_every_option_and_applies_it() {
    if which("ssh").is_none() {
        return; // no ssh client on this machine
    }
    let d = tempfile::tempdir().unwrap();
    let key = hub_key(d.path());
    let h = cfg(
        "gpu-box-2",
        "x",
        Some("ops@10.0.0.5"),
        Some(2222),
        false,
        true,
    );
    let argv = ssh_argv(&h, &key, REMOTE_CMD).unwrap();
    let done = std::process::Command::new(&argv[0])
        .arg("-G")
        .args(&argv[1..])
        .env_clear()
        .envs(ssh_env())
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert!(
        done.status.success(),
        "{}",
        String::from_utf8_lossy(&done.stderr)
    );
    let out = String::from_utf8_lossy(&done.stdout);
    let lines: Vec<&str> = out.lines().collect();
    let c: HashMap<&str, &str> = lines.iter().filter_map(|l| l.split_once(' ')).collect();
    assert!(
        c["batchmode"] == "yes"
            && c["stricthostkeychecking"] == "true"
            && c["updatehostkeys"] == "false"
    );
    assert!(
        c["identitiesonly"] == "yes"
            && c["forwardagent"] == "no"
            && c["clearallforwardings"] == "yes"
    );
    assert!(c["port"] == "2222" && c["user"] == "ops" && c["hostname"] == "10.0.0.5");
    assert!(
        c["connecttimeout"] == "5"
            && c["serveraliveinterval"] == "5"
            && c["serveralivecountmax"] == "3"
    );
    assert!(c["passwordauthentication"] == "no" && c["kbdinteractiveauthentication"] == "no");
    assert!(
        c["identityagent"] == "none"
            && c["controlmaster"] == "false"
            && !c.contains_key("controlpath")
    );
    assert!(
        c["requesttty"] == "false" && c["escapechar"] == "none" && c["permitlocalcommand"] == "no"
    );
    assert!(c["verifyhostkeydns"] == "false" && c["forwardx11"] == "no");
    // the hub's key is the ONLY identity ssh will offer: no default identity files are added next to it
    let ids: Vec<&&str> = lines
        .iter()
        .filter(|l| l.starts_with("identityfile "))
        .collect();
    assert_eq!(ids, [&format!("identityfile {}", key.display()).as_str()]);
}

/// Where the hub would find `prog`: its fixed tool path, never this process's PATH (tools.rs).
fn which(prog: &str) -> Option<PathBuf> {
    ec_hub::tools::resolve(prog)
}

// ── environments ───────────────────────────────────────────────────────────────────────────────────────
fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let m: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    move |k: &str| m.get(k).cloned()
}

#[test]
fn test_ssh_env_is_scrubbed() {
    let env = env_of(&[
        ("SSH_AUTH_SOCK", "/run/agent"),
        ("HF_TOKEN", "hf_secretsecret"),
        ("HTTPS_PROXY", "http://p"),
        ("DISPLAY", ":0"),
        ("SSH_ASKPASS", "/bin/askpass"),
        ("LANG", "de_DE.UTF-8"),
        ("USER", "ops"),
        ("HOME", "/home/ops"),
        ("PATH", "/usr/bin:/bin"),
    ]);
    let got: Vec<(String, String)> = ssh_env_from(&env);
    // PATH is the hub's fixed tool path, not the caller's (hardening deviation; Python passes PATH through)
    let want: Vec<(String, String)> = [
        ("HOME", "/home/ops"),
        ("PATH", ec_hub::tools::SAFE_PATH),
        ("LANG", "C.UTF-8"),
        ("USER", "ops"),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
    .collect();
    assert_eq!(got, want);
}

/// Neither ssh's nor the local node's environment ever carries the caller's PATH: a writable directory first on it
/// (`~/bin`, `.`) must not decide which `ssh` or `docker` runs. Only the explicit test/development override
/// `ENGINE_CONSOLE_TOOL_PATH` replaces the fixed path, and only its absolute entries count (hardening finding 6).
#[test]
fn test_children_get_the_tool_path_never_the_callers_path() {
    let path = |v: &[(String, String)]| v.iter().find(|(k, _)| k == "PATH").map(|(_, p)| p.clone());
    let caller = env_of(&[("PATH", "/home/ops/bin:.:/usr/bin"), ("HOME", "/home/ops")]);
    assert_eq!(
        path(&ssh_env_from(&caller)).as_deref(),
        Some(ec_hub::tools::SAFE_PATH)
    );
    assert_eq!(
        path(&local_env_from(&caller)).as_deref(),
        Some(ec_hub::tools::SAFE_PATH)
    );
    let fakes = env_of(&[
        ("PATH", "/home/ops/bin"),
        ("ENGINE_CONSOLE_TOOL_PATH", "/opt/fakes:bin:/usr/bin"),
    ]);
    assert_eq!(
        path(&ssh_env_from(&fakes)).as_deref(),
        Some("/opt/fakes:/usr/bin")
    );
    assert_eq!(
        path(&local_env_from(&fakes)).as_deref(),
        Some("/opt/fakes:/usr/bin")
    );
}

#[test]
fn test_ssh_env_falls_back_when_the_environment_is_bare() {
    let env = ssh_env_from(&env_of(&[("HF_TOKEN", "x")]));
    let m: HashMap<&str, &str> = env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    assert!(!m["PATH"].is_empty() && !m["HOME"].is_empty() && m["LANG"] == "C.UTF-8");
    assert!(
        m.keys()
            .all(|k| ["HOME", "PATH", "LANG", "USER"].contains(k))
    );
}

#[test]
fn test_local_env_never_carries_secrets() {
    let env = local_env_from(&env_of(&[
        ("HF_TOKEN", "hf_secretsecret"),
        ("VLLM_API_KEY", "k"),
        ("ENGINE_CONSOLE_ENV_FILE", "/x"),
        ("SSH_AUTH_SOCK", "/a"),
        ("HOME", "/home/ops"),
        ("PATH", "/usr/bin"),
        ("USER", "ops"),
        ("XDG_RUNTIME_DIR", "/run/user/1000"),
        ("DOCKER_HOST", "unix:///run/user/1000/docker.sock"),
    ]));
    let m: HashMap<&str, &str> = env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    assert!(m.keys().all(|k| LOCAL_ENV_KEYS.contains(k)));
    assert_eq!(m["DOCKER_HOST"], "unix:///run/user/1000/docker.sock");
    assert_eq!(m["XDG_RUNTIME_DIR"], "/run/user/1000");
    assert_eq!(m["PATH"], ec_hub::tools::SAFE_PATH); // never the caller's PATH
    for k in [
        "HF_TOKEN",
        "VLLM_API_KEY",
        "SSH_AUTH_SOCK",
        "ENGINE_CONSOLE_ENV_FILE",
    ] {
        assert!(!m.contains_key(k));
    }
}

/// D8: the default local node is the `ecnode` binary next to the running executable (Python: `python -I -m ecnode`).
#[test]
fn test_local_argv() {
    let d = tempfile::tempdir().unwrap();
    let base = Settings::defaults();
    let default = vec![
        sibling_ecnode().to_string_lossy().into_owned(),
        "--stdio".to_string(),
    ];
    assert_eq!(local_argv(&base), default);
    let mut with_policy = base.clone();
    with_policy.node_policy = Some(d.path().join("policy.json"));
    let mut want = default.clone();
    want.extend([
        "--policy".to_string(),
        d.path().join("policy.json").display().to_string(),
    ]);
    assert_eq!(local_argv(&with_policy), want);
    // an override is the complete command line: the policy setting is not appended to it
    let mut over = with_policy;
    over.node_command = Some(vec!["python3".into(), "fake.py".into()]);
    assert_eq!(local_argv(&over), ["python3", "fake.py"]);
    let mut empty = base;
    empty.node_command = Some(Vec::new());
    assert_eq!(local_argv(&empty), default);
}

// ── the line that authorizes the hub's key on a machine ────────────────────────────────────────────────
#[test]
fn test_key_line_format() {
    let line = key_line(PUB, Some("10.0.0.5"), REMOTE_CMD).unwrap();
    assert_eq!(
        line,
        "restrict,from=\"10.0.0.5\",command=\"/usr/local/lib/ecnode/ecnode --stdio\" ssh-ed25519 \
         AAAAC3NzaC1lZDI1NTE5AAAAIEZBS0UtVEVTVC1LRVktTk9ULUEtUkVBTC1PTkUtMDAw engine-console-hub"
    );
    assert!(!line.contains("user@somewhere")); // the public key's own comment is replaced, never kept
}

#[test]
fn test_key_line_defaults_and_from_variants() {
    assert!(
        key_line(&format!("{PUB}\n"), None, DEFAULT_REMOTE_COMMAND)
            .unwrap()
            .starts_with("restrict,command=\"/usr/local/lib/ecnode/ecnode --stdio\" ssh-ed25519 ")
    );
    for ok in [
        "10.0.0.0/24",
        "10.0.0.5,10.0.0.6",
        "fd00::1",
        "*.lab.example",
        "!10.0.0.9,10.0.0.*",
    ] {
        assert!(
            key_line(PUB, Some(ok), DEFAULT_REMOTE_COMMAND)
                .unwrap()
                .contains(&format!("from=\"{ok}\""))
        );
    }
}

#[test]
fn test_key_line_refuses_anything_but_one_public_key() {
    let b64 = PUB.split_whitespace().nth(1).unwrap();
    let cases = [
        String::new(),
        "ssh-ed25519".into(),
        "garbage AAAAC3NzaC1lZDI1NTE5AAAAIOMq".into(),
        "ssh-dss AAAAC3NzaC1lZDI1NTE5AAAAIOMq".into(),
        "ssh-ed25519 not*base64*at*all*!!".into(),
        "ssh-ed25519 AAAA".into(),
        format!("{PUB}\nssh-ed25519 {b64}"),
        format!("command=\"x\" {PUB}"),
        "ssh-ed25519\rAAAAC3NzaC1lZDI1NTE5AAAAIOMq\nrest".into(),
    ];
    for p in cases {
        assert!(
            key_line(&p, Some("10.0.0.5"), DEFAULT_REMOTE_COMMAND).is_err(),
            "{p:?}"
        );
    }
}

#[test]
fn test_key_line_refuses_a_from_that_could_leave_its_quotes() {
    let long = "a".repeat(257);
    for f in [
        "",
        "10.0.0.5\"",
        "10.0.0.5 ",
        "10.0.0.5\n",
        "a\",command=\"x",
        long.as_str(),
        "10.0.0.5;x",
    ] {
        let e = key_line(PUB, Some(f), DEFAULT_REMOTE_COMMAND).unwrap_err();
        assert!(err_text(&e).contains("--from"), "{f:?}");
    }
}

#[test]
fn test_key_line_refuses_a_remote_command_that_could_leave_its_quotes() {
    for c in [
        "/x\"; reboot; \"",
        "/x\n",
        "relative/ecnode --stdio",
        "/x ; id",
        "/x --a=$(id)",
    ] {
        let e = key_line(PUB, Some("10.0.0.5"), c).unwrap_err();
        assert!(err_text(&e).contains("remote node command"), "{c:?}");
    }
}

// ── init_key (real ssh-keygen in a temp dir) ───────────────────────────────────────────────────────────
fn has_keygen() -> bool {
    which("ssh-keygen").is_some()
}

fn mode(p: &Path) -> u32 {
    std::fs::metadata(p).unwrap().permissions().mode() & 0o777
}

#[test]
fn test_init_key_creates_a_private_key_and_returns_the_line() {
    if !has_keygen() {
        return;
    }
    let _u = UMASK.lock();
    let d = tempfile::tempdir().unwrap();
    let key = d.path().join("config").join("ssh").join("hub-key");
    let line = init_key(&key, Some("10.0.0.5"), REMOTE_CMD).unwrap();
    assert!(line.starts_with(
        "restrict,from=\"10.0.0.5\",command=\"/usr/local/lib/ecnode/ecnode --stdio\" ssh-ed25519 "
    ));
    assert!(line.ends_with(" engine-console-hub"));
    assert_eq!(mode(&key), 0o600);
    assert_eq!(mode(key.parent().unwrap()), 0o700);
    assert_eq!(mode(key.parent().unwrap().parent().unwrap()), 0o700);
    assert!(!line.contains("PRIVATE KEY"));
    // a throwaway key made by this test in its own temp dir
    assert!(
        std::fs::read_to_string(&key)
            .unwrap()
            .starts_with("-----BEGIN OPENSSH PRIVATE KEY-----")
    );
    let pubtext =
        std::fs::read_to_string(d.path().join("config").join("ssh").join("hub-key.pub")).unwrap();
    assert!(line.contains(pubtext.split_whitespace().nth(1).unwrap()));
    // the key is passphrase-less (it is used unattended): ssh-keygen can derive its public half without a prompt
    let derived = std::process::Command::new(which("ssh-keygen").unwrap())
        .args(["-y", "-f"])
        .arg(&key)
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert!(derived.status.success());
    assert!(
        line.contains(
            String::from_utf8_lossy(&derived.stdout)
                .split_whitespace()
                .nth(1)
                .unwrap()
        )
    );
}

#[test]
fn test_init_key_never_overwrites_an_existing_key() {
    if !has_keygen() {
        return;
    }
    let _u = UMASK.lock();
    let d = tempfile::tempdir().unwrap();
    let key = d.path().join("ssh").join("hub-key");
    let pubp = d.path().join("ssh").join("hub-key.pub");
    let first = init_key(&key, Some("10.0.0.5"), REMOTE_CMD).unwrap();
    let (before, before_pub) = (std::fs::read(&key).unwrap(), std::fs::read(&pubp).unwrap());
    let again = init_key(&key, Some("10.0.0.5"), REMOTE_CMD).unwrap();
    assert_eq!(again, first);
    assert_eq!(std::fs::read(&key).unwrap(), before);
    assert_eq!(std::fs::read(&pubp).unwrap(), before_pub);
    assert_ne!(init_key(&key, Some("10.0.0.9"), REMOTE_CMD).unwrap(), first); // same key, a different from=
    assert_eq!(std::fs::read(&key).unwrap(), before);
}

#[test]
fn test_init_key_rederives_a_lost_public_half() {
    if !has_keygen() {
        return;
    }
    let _u = UMASK.lock();
    let d = tempfile::tempdir().unwrap();
    let key = d.path().join("hub-key");
    let first = init_key(&key, Some("10.0.0.5"), DEFAULT_REMOTE_COMMAND).unwrap();
    std::fs::remove_file(d.path().join("hub-key.pub")).unwrap();
    assert_eq!(
        init_key(&key, Some("10.0.0.5"), DEFAULT_REMOTE_COMMAND).unwrap(),
        first
    );
}

#[test]
fn test_init_key_validates_before_touching_the_disk() {
    let _u = UMASK.lock();
    let d = tempfile::tempdir().unwrap();
    let key = d.path().join("new-dir").join("hub-key");
    let e = init_key(&key, Some("bad\"from"), DEFAULT_REMOTE_COMMAND).unwrap_err();
    assert!(err_text(&e).contains("--from"));
    assert!(!d.path().join("new-dir").exists());
}

#[test]
fn test_init_key_does_not_clobber_a_stray_public_key_or_follow_a_dangling_link() {
    let _u = UMASK.lock();
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("hub-key.pub"), "precious\n").unwrap();
    let e = init_key(
        &d.path().join("hub-key"),
        Some("10.0.0.5"),
        DEFAULT_REMOTE_COMMAND,
    )
    .unwrap_err();
    assert!(err_text(&e).contains("without its private key"));
    assert_eq!(
        std::fs::read_to_string(d.path().join("hub-key.pub")).unwrap(),
        "precious\n"
    );
    let link = d.path().join("linked");
    std::os::unix::fs::symlink(d.path().join("nowhere"), &link).unwrap();
    let e = init_key(&link, Some("10.0.0.5"), DEFAULT_REMOTE_COMMAND).unwrap_err();
    assert!(err_text(&e).contains("dangling"));
    assert!(!d.path().join("nowhere").exists());
}

#[test]
fn test_init_key_without_ssh_keygen_is_a_clear_error() {
    let _u = UMASK.lock();
    let d = tempfile::tempdir().unwrap();
    let e = init_key_with(
        &d.path().join("k"),
        Some("10.0.0.5"),
        DEFAULT_REMOTE_COMMAND,
        "definitely-not-an-installed-program",
        30.0,
    )
    .unwrap_err();
    assert!(err_text(&e).contains("ssh-keygen was not found"));
    assert!(!d.path().join("k").exists());
}

#[test]
fn test_init_key_restores_the_umask() {
    use nix::sys::stat::{Mode, umask};
    let _u = UMASK.lock();
    let d = tempfile::tempdir().unwrap();
    let old = umask(Mode::from_bits_truncate(0o022));
    let r = init_key_with(
        &d.path().join("k"),
        Some("10.0.0.5"),
        DEFAULT_REMOTE_COMMAND,
        "definitely-not-an-installed-program",
        30.0,
    );
    let seen = umask(Mode::from_bits_truncate(0o022));
    umask(old);
    assert!(r.is_err());
    assert_eq!(seen.bits(), 0o022);
}
