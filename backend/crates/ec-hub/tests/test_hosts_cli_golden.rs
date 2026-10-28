//! `engine-console hosts ...` byte for byte against the Python reference (golden transcripts).
//!
//! `tests/data/hosts_cli_golden.json` is written by `backend/parity/gen/hosts_cli_golden.py` from the Python CLI; every
//! case is replayed here with the Rust binary in a fresh temp dir with the same environment built from scratch (HOME,
//! data dir, hosts.json and hub key inside it; PATH = the case's fake bin dir, then the parity fakes, then
//! /usr/bin:/bin). A remote host is reached through the case's fake `ssh` (first on PATH; its address is a
//! `.invalid` name); `ssh-keygen` is the real one, writing a throwaway key into the temp dir. Exit code, stdout and
//! stderr must match after the same normalisation as the generator (temp dir -> `<T>`, key base64 -> `<PUBKEY>`,
//! streams cut after a marker where PORTING.md §14 documents a deviation). This covers the cutover contract of
//! PORTING.md §18 (the one `restrict,` line) and the link warnings `hosts check` prints on stderr.

#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

const GOLDEN: &str = include_str!("data/hosts_cli_golden.json");
const BIN: &str = env!("CARGO_BIN_EXE_engine-console");
const FAKES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/fakes/bin");

#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    code: i64,
    stdout: String,
    stderr: String,
}

fn normalise(text: &str, tmp: &str) -> String {
    let re = regex::Regex::new(r"(ssh-ed25519 )[A-Za-z0-9+/]+={0,3}").unwrap();
    re.replace_all(&text.replace(tmp, "<T>"), "${1}<PUBKEY>")
        .into_owned()
}

fn case_env(tmp: &str, extra: Option<&Value>) -> BTreeMap<String, String> {
    let mut env: BTreeMap<String, String> = [
        ("HOME", format!("{tmp}/home")),
        ("LANG", "C.UTF-8".into()),
        ("PATH", format!("{tmp}/bin:{FAKES}:/usr/bin:/bin")),
        // the Rust hub finds ssh / ssh-keygen on its tool path, never on PATH (tools.rs); Python reads PATH. Same
        // directories for both, so the transcripts compare as they are.
        (
            "ENGINE_CONSOLE_TOOL_PATH",
            format!("{tmp}/bin:{FAKES}:/usr/bin:/bin"),
        ),
        ("XDG_CONFIG_HOME", format!("{tmp}/xdg-config")),
        ("XDG_DATA_HOME", format!("{tmp}/xdg-data")),
        ("XDG_STATE_HOME", format!("{tmp}/xdg-state")),
        ("XDG_CACHE_HOME", format!("{tmp}/xdg-cache")),
        ("XDG_RUNTIME_DIR", format!("{tmp}/xdg-run")),
        ("ENGINE_CONSOLE_ENV_FILE", format!("{tmp}/no-env-file")),
        ("ENGINE_CONSOLE_DATA_DIR", format!("{tmp}/data")),
        ("HOSTS_FILE", format!("{tmp}/hosts.json")),
        ("HUB_KEY_FILE", format!("{tmp}/ssh/hub-key")),
        ("HF_ENDPOINT", "https://127.0.0.1:9".into()),
        ("OTLP_ENABLED", "false".into()),
        ("OTLP_ENDPOINT", "127.0.0.1:9".into()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    if let Some(Value::Object(m)) = extra {
        for (k, v) in m {
            env.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
        }
    }
    env
}

fn run(argv: &[String], tmp: &str, env: &BTreeMap<String, String>) -> Outcome {
    let mut child = Command::new(BIN)
        .arg("hosts")
        .args(argv.iter().map(|a| a.replace("<T>", tmp)))
        .current_dir(format!("{tmp}/home"))
        .env_clear()
        .envs(env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("engine-console hosts {argv:?} did not finish in 60 s");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let out = child.wait_with_output().unwrap();
    Outcome {
        code: i64::from(out.status.code().unwrap_or(-1)),
        stdout: normalise(&String::from_utf8_lossy(&out.stdout), tmp),
        stderr: normalise(&String::from_utf8_lossy(&out.stderr), tmp),
    }
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|x| x.as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn replay(case: &Value) -> Result<(), String> {
    let name = case["name"].as_str().unwrap_or("?");
    let dir = tempfile::tempdir().unwrap();
    let tmp = dir.path().to_str().unwrap().to_string();
    for d in ["home", "bin", "data"] {
        let p = dir.path().join(d);
        std::fs::create_dir_all(&p).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    if let Some(Value::Object(files)) = case.get("files") {
        for (rel, f) in files {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap_or(Path::new("/"))).unwrap();
            std::fs::write(&p, f["content"].as_str().unwrap_or_default()).unwrap();
            let mode = u32::try_from(f["mode"].as_u64().unwrap_or(0o600)).unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
        }
    }
    let env = case_env(&tmp, case.get("env"));
    for s in case["setup"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
        let r = run(&strings(s), &tmp, &env);
        if r.code != 0 {
            return Err(format!("{name}: setup {s} failed: {r:?}"));
        }
    }
    let mut got = run(&strings(&case["argv"]), &tmp, &env);
    if let Some(Value::Object(cut)) = case.get("cut") {
        for (stream, marker) in cut {
            let marker = marker.as_str().unwrap_or_default();
            let text = if stream == "stdout" {
                &mut got.stdout
            } else {
                &mut got.stderr
            };
            match text.find(marker) {
                Some(i) => text.truncate(i + marker.len()),
                None => {
                    return Err(format!(
                        "{name}: marker {marker:?} not in {stream}: {text:?}"
                    ));
                }
            }
        }
    }
    let e = &case["expect"];
    let want = Outcome {
        code: e["code"].as_i64().unwrap_or(-99),
        stdout: e["stdout"].as_str().unwrap_or_default().to_string(),
        stderr: e["stderr"].as_str().unwrap_or_default().to_string(),
    };
    if got == want {
        Ok(())
    } else {
        Err(format!("{name}:\n  python: {want:?}\n  rust:   {got:?}"))
    }
}

#[test]
fn test_hosts_cli_matches_the_python_golden_transcripts() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(cases.len() >= 20, "the golden file lost its cases");
    let failures: Vec<String> = cases.iter().filter_map(|c| replay(c).err()).collect();
    assert!(
        failures.is_empty(),
        "{} of {} cases differ from Python:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn test_the_init_key_golden_prints_exactly_one_restrict_line() {
    // the fleet installer keeps the first stdout line starting `restrict,` (PORTING.md §18): exactly one per run
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let mut seen = 0;
    for c in doc["cases"].as_array().unwrap() {
        let argv = strings(&c["argv"]);
        let help = argv.iter().any(|a| a == "-h" || a == "--help");
        if argv.first().map(String::as_str) == Some("init-key") && !help && c["expect"]["code"] == 0
        {
            let out = c["expect"]["stdout"].as_str().unwrap();
            assert_eq!(
                out.lines().filter(|l| l.starts_with("restrict,")).count(),
                1,
                "{out}"
            );
            seen += 1;
        }
    }
    assert!(seen >= 4);
}
