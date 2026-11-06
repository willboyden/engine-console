//! The Config files page's two routes (beyond Python): `GET /config/{file}` serves only what may be shown, through the
//! loaders' own file checks, and `POST /config/validate` runs the real loaders on a draft without writing anything,
//! echoing a value or storing the draft in the audit log.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/kit.rs"]
mod kit;

use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;

use ec_hub::services::{config_check, hosts};
use kit::TestEnv;
use serde_json::{Value, json};

const P: &str = "/api/v1";

fn write_mode(path: &Path, text: &str, mode: u32) {
    std::fs::write(path, text).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

async fn check(env: &mut TestEnv, file: &str, text: &str) -> Value {
    let r = env
        .post(&format!("{P}/config/validate"))
        .json(&json!({"file": file, "text": text}))
        .send()
        .await;
    assert_eq!(r.status, 200, "{}", r.text());
    r.json()
}

/// Every file name in the tree under `dir`, with its size and modification time: proof that nothing was written.
fn snapshot(dir: &Path) -> Vec<(String, u64, std::time::SystemTime)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let m = e.metadata().unwrap();
            if m.is_dir() {
                stack.push(e.path());
            } else {
                out.push((
                    e.path().display().to_string(),
                    m.len(),
                    m.modified().unwrap(),
                ));
            }
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn test_hosts_is_served_only_when_the_loader_accepts_it() {
    let mut env = TestEnv::new();
    let path = env.cfg.hosts_file.clone();
    // no file: nothing to serve, the template is there
    let r = env.get(&format!("{P}/config/hosts")).send().await.json();
    assert_eq!(
        (&r["exists"], &r["served"], &r["reason"]),
        (&json!(false), &json!(false), &json!("missing"))
    );
    assert!(hosts::parse_hosts_bytes(r["templates"]["hosts"].as_str().unwrap().as_bytes()).is_ok());
    // a valid 0600 file: served as it is
    let good = "{\"hosts\": [{\"id\": \"local\", \"label\": \"Work\"},\n {\"id\": \"b\", \"ssh\": \"ops@192.0.2.9\"}]}\n";
    write_mode(&path, good, 0o600);
    let r = env.get(&format!("{P}/config/hosts")).send().await.json();
    assert_eq!(
        (&r["served"], r["text"].as_str()),
        (&json!(true), Some(good))
    );
    // group-writable: refused by the loader's own check, not served
    write_mode(&path, good, 0o620);
    let r = env.get(&format!("{P}/config/hosts")).send().await.json();
    assert_eq!(
        (&r["served"], &r["reason"], &r["text"]),
        (&json!(false), &json!("refused"), &Value::Null)
    );
    assert!(
        r["problem"]
            .as_str()
            .unwrap()
            .contains("must not be group- or world-writable")
    );
    // invalid: the loader's message, no text
    write_mode(
        &path,
        "{\"hosts\": [{\"id\": \"local\", \"oops\": 1}]}",
        0o600,
    );
    let r = env.get(&format!("{P}/config/hosts")).send().await.json();
    assert_eq!(
        (&r["served"], &r["reason"], &r["text"]),
        (&json!(false), &json!("invalid"), &Value::Null)
    );
    assert!(
        r["problem"]
            .as_str()
            .unwrap()
            .contains("unknown key(s) oops")
    );
    // valid, but something in it looks like a secret: not served
    let tokened = "{\"hosts\": [{\"id\": \"local\", \"label\": \"hf_abcdefghijklmnop\"}]}";
    write_mode(&path, tokened, 0o600);
    let r = env.get(&format!("{P}/config/hosts")).send().await;
    assert!(!r.text().contains("hf_abcdefghijklmnop"));
    assert_eq!(r.json()["reason"], json!("secret"));
    // a symlink is refused (O_NOFOLLOW), and the target is never shown
    std::fs::remove_file(&path).unwrap();
    let real = env.tmp.path().join("real-hosts.json");
    write_mode(&real, good, 0o600);
    std::os::unix::fs::symlink(&real, &path).unwrap();
    let r = env.get(&format!("{P}/config/hosts")).send().await.json();
    assert_eq!(
        (&r["served"], &r["reason"]),
        (&json!(false), &json!("refused"))
    );
}

#[tokio::test]
async fn test_the_env_file_is_never_served_and_policy_is_template_only() {
    let mut env = TestEnv::new();
    let r = env.get(&format!("{P}/config/env")).send().await.json();
    assert_eq!(
        (&r["served"], &r["text"], &r["reason"]),
        (&json!(false), &Value::Null, &json!("secret"))
    );
    let names: Vec<&str> = r["fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ec_hub::config::SETTINGS
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
    );
    // even an existing env file with a token in it: only its path and that it exists
    let f = env.tmp.path().join("env");
    write_mode(&f, "HF_TOKEN=hf_abcdefghijklmnopqrstu\n", 0o600);
    let v = config_check::current_as("env", &env.cfg, &f, 0).unwrap();
    let s = v.to_string();
    assert!(
        !s.contains("hf_abcdefghijklmnopqrstu") && v["exists"] == json!(true),
        "{s}"
    );

    let r = env.get(&format!("{P}/config/policy")).send().await.json();
    assert_eq!((&r["served"], &r["path"]), (&json!(false), &Value::Null));
    assert_eq!(
        r["templates"]["monitor"].as_str().unwrap(),
        include_str!("../../../../node/examples/policy.monitor.json")
    );
    assert_eq!(
        r["templates"]["manage"].as_str().unwrap(),
        include_str!("../../../../node/examples/policy.manage.json")
    );

    let r = env.get(&format!("{P}/config/secrets")).send().await;
    assert_eq!((r.status, r.code().as_str()), (404, "not_found"));
    // a write method on the read route does not exist
    let r = env
        .put(&format!("{P}/config/hosts"))
        .json(&json!({"text": "{}"}))
        .send()
        .await;
    assert_eq!(r.status, 405);
}

#[tokio::test]
async fn test_hosts_validation_is_the_loader() {
    let mut env = TestEnv::new();
    let path = env.tmp.path().join("hosts-corpus.json");
    let corpus = [
        "{\"hosts\": [{\"id\": \"local\"}]}",
        "{\"hosts\": [{\"id\": \"local\"}, {\"id\": \"gpu-2\", \"ssh\": \"ops@192.0.2.5\", \"port\": 2222}]}",
        "{\"hosts\": []}",
        "{\"hosts\": [{\"id\": \"local\", \"id\": \"x\"}]}",
        "{\"hosts\": [{\"id\": \"local\", \"port\": NaN}]}",
        "{\"hosts\": [{\"id\": \"Bad\"}]}",
        "{\"hosts\": [{\"id\": \"a\", \"ssh\": \"-oProxyCommand=x\"}]}",
        "{\"hosts\": [{\"id\": \"local\", \"ssh\": \"x\"}]}",
        "{\"hosts\": [{\"id\": \"a\", \"ssh\": \"h\"}, {\"id\": \"a\", \"ssh\": \"h\"}]}",
        "{\"hosts\": [{\"id\": \"a\", \"ssh\": \"h\", \"port\": 70000}]}",
        "{\"hosts\": [{\"id\": \"a\", \"ssh\": \"h\", \"enabled\": \"yes\"}]}",
        "{\"hosts\": [{\"id\": \"local\"}],\n \"extra\": 1}",
        "{\"hosts\": [\n  {\"id\": \"local\"},\n  {\"id\": \"b\" \"ssh\": \"h\"}\n]}",
        "not json",
        "",
    ];
    for text in corpus {
        write_mode(&path, text, 0o600);
        let want = hosts::load_hosts(&path);
        let got = check(&mut env, "hosts", text).await;
        match want {
            Ok(list) => {
                assert_eq!(got["ok"], json!(true), "{text}: {got}");
                assert_eq!(
                    got["outline"]["hosts"].as_array().unwrap().len(),
                    list.len()
                );
            }
            Err(e) => {
                assert_eq!(got["ok"], json!(false), "{text}");
                let prefix = format!("{}: ", path.display());
                let msg = e.to_string();
                assert_eq!(
                    got["errors"][0]["message"].as_str().unwrap(),
                    msg.strip_prefix(&prefix).unwrap(),
                    "{text}"
                );
            }
        }
    }
    // the syntax error on line 3 is located
    let r = check(&mut env, "hosts", corpus[12]).await;
    assert_eq!(r["errors"][0]["line"], json!(3), "{r}");
    // no "local": a warning, still valid
    let r = check(
        &mut env,
        "hosts",
        "{\"hosts\": [{\"id\": \"a\", \"ssh\": \"h\"}]}",
    )
    .await;
    assert_eq!(r["ok"], json!(true));
    assert!(
        r["warnings"][0]["message"]
            .as_str()
            .unwrap()
            .contains("local")
    );
}

#[tokio::test]
async fn test_env_validation_is_the_settings_loader_and_never_echoes_a_value() {
    let mut env = TestEnv::new();
    let good = "# comment\nEGRESS_PROXY=http://127.0.0.1:8082\nMAX_SSE_PER_KEY=3\nexport OTLP_ENABLED=no\n";
    let r = check(&mut env, "env", good).await;
    assert_eq!(r["ok"], json!(true), "{r}");
    assert_eq!(
        r["outline"]["set"],
        json!(["EGRESS_PROXY", "MAX_SSE_PER_KEY", "OTLP_ENABLED"])
    );
    // a value that does not parse: the field and its line, never the value
    let r = check(
        &mut env,
        "env",
        "A_NOTE=1\nENGINE_CONSOLE_PORT=99999-secret-ish\n",
    )
    .await;
    assert_eq!(r["ok"], json!(false));
    assert_eq!(r["errors"][0]["field"], json!("port"));
    assert_eq!(r["errors"][0]["line"], json!(2));
    assert!(
        r["errors"][0]["message"]
            .as_str()
            .unwrap()
            .starts_with("ENGINE_CONSOLE_PORT: ")
    );
    assert!(!r.to_string().contains("99999-secret-ish"));
    let r = check(&mut env, "env", "ENGINE_CONSOLE_PORT=70000\n").await;
    assert_eq!(
        r["errors"][0]["message"],
        json!("ENGINE_CONSOLE_PORT: port must be 0-65535")
    );
    // the same answer as the real loader on a file with this text
    let f = env.tmp.path().join("envfile");
    write_mode(&f, "HF_CACHE_DIR=relative/dir\n", 0o600);
    let want = ec_hub::config::Settings::load(&[], Some(&f)).unwrap_err();
    let r = check(&mut env, "env", "HF_CACHE_DIR=relative/dir\n").await;
    assert_eq!(
        r["errors"][0]["message"].as_str().unwrap(),
        format!("HF_CACHE_DIR: {}", want.message)
    );
    // warnings: a bare generic name, an unknown name, a setting with no effect, a token, trust_loopback
    let text = "PORT=1\nNOT_A_SETTING=x\nDOCKER_BIN=docker\nHF_TOKEN=hf_abcdefghijklmnopqrstu\nENGINE_CONSOLE_TRUST_LOOPBACK=true\n";
    let r = check(&mut env, "env", text).await;
    assert_eq!(r["ok"], json!(true), "{r}");
    let all = r.to_string();
    assert!(!all.contains("hf_abcdefghijklmnopqrstu"), "{all}");
    let lines: Vec<u64> = r["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|w| w["line"].as_u64())
        .collect();
    assert_eq!(lines, vec![1, 2, 3, 4, 5], "{r}");
    assert_eq!(r["outline"]["unknown"], json!(["NOT_A_SETTING"]));
    // over the loader's cap: it would ignore the whole file (a draft that big is over the request body cap too)
    let big = "# x\n".repeat(300_000);
    let r = config_check::validate("env", &big).unwrap();
    assert_eq!(r["ok"], json!(false));
    assert!(
        r["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("ignore it")
    );
    let r = env
        .post(&format!("{P}/config/validate"))
        .json(&json!({"file": "env", "text": big}))
        .send()
        .await;
    assert_eq!(r.status, 413);
}

#[tokio::test]
async fn test_policy_validation_accepts_the_shipped_examples() {
    let mut env = TestEnv::new();
    for (name, text) in [
        (
            "monitor",
            include_str!("../../../../node/examples/policy.monitor.json"),
        ),
        (
            "manage",
            include_str!("../../../../node/examples/policy.manage.json"),
        ),
    ] {
        let r = check(&mut env, "policy", text).await;
        assert_eq!(r["ok"], json!(true), "{name}: {r}");
        assert_eq!(r["outline"]["mode"], json!(name));
    }
    let r = check(
        &mut env,
        "policy",
        "{\"version\": 1, \"allow_delet\": true}",
    )
    .await;
    assert_eq!(r["ok"], json!(false));
    assert!(
        r["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("unknown key 'allow_delet'")
    );
}

#[tokio::test]
async fn test_validation_writes_nothing_and_the_audit_keeps_metadata_only() {
    let mut env = TestEnv::new();
    let before = snapshot(env.tmp.path());
    let secret = "EGRESS_PROXY=http://user:VERYSECRET-PASSWORD@127.0.0.1:8082\n";
    let r = check(&mut env, "env", secret).await;
    assert_eq!(r["ok"], json!(true));
    check(&mut env, "hosts", "{\"hosts\": [{\"id\": \"local\"}]}").await;
    let after = snapshot(env.tmp.path());
    // the database may have grown (the audit row); no other file appeared or changed
    let strip = |v: Vec<(String, u64, std::time::SystemTime)>| -> Vec<(String, u64, std::time::SystemTime)> {
        v.into_iter()
            .filter(|(p, _, _)| !p.contains("console.db"))
            .collect()
    };
    assert_eq!(strip(before), strip(after));
    let c = env.container();
    let rows = c
        .store
        .all(
            "SELECT params FROM audit WHERE method='POST' AND path=?",
            &[&format!("{P}/config/validate")],
        )
        .unwrap();
    assert_eq!(rows.len(), 2);
    for row in rows {
        let p = row.text("params").unwrap();
        assert!(
            !p.contains("VERYSECRET") && !p.contains("EGRESS_PROXY") && p.contains("bytes"),
            "{p}"
        );
    }
}

#[tokio::test]
async fn test_a_malformed_request_is_422() {
    let mut env = TestEnv::new();
    for body in [
        json!({"file": "hosts"}),
        json!({"text": "x"}),
        json!({"file": "secrets", "text": "x"}),
        json!({"file": "env", "text": 5}),
    ] {
        let r = env
            .post(&format!("{P}/config/validate"))
            .json(&body)
            .send()
            .await;
        assert_eq!(
            (r.status, r.code().as_str()),
            (422, "validation_error"),
            "{body}"
        );
    }
}

#[tokio::test]
async fn test_a_large_env_draft_is_checked_in_linear_time_with_capped_diagnostics() {
    let mut env = TestEnv::new();
    // the most lines checked, each a distinct unknown name of ~100 characters with an expansion: about 1 MiB, every
    // line a warning. The old per-name rescans made this quadratic.
    let n = config_check::MAX_ENV_DRAFT_LINES;
    let pad = "N".repeat(100);
    let text: String = (0..n)
        .map(|i| format!("X{i:05}_{pad}=${{X{:05}_{pad}:-v}}\n", i.saturating_sub(1)))
        .collect();
    assert!(text.len() < ec_hub::config::MAX_ENV_FILE);
    let t0 = std::time::Instant::now();
    let r = check(&mut env, "env", &text).await;
    let took = t0.elapsed();
    assert!(took < std::time::Duration::from_secs(5), "{took:?}");
    assert_eq!(r["ok"], json!(true));
    let warnings = r["warnings"].as_array().unwrap();
    assert_eq!(warnings.len(), config_check::MAX_DIAGNOSTICS + 1);
    let last = warnings.last().unwrap()["message"].as_str().unwrap();
    assert!(last.contains("more not shown"), "{last}");
    // names in messages and in the outline are cut short
    let first = warnings[0]["message"].as_str().unwrap();
    assert!(
        first.chars().count() < 150 && first.contains('…'),
        "{first}"
    );
    let unknown = r["outline"]["unknown"].as_array().unwrap();
    assert_eq!(unknown.len(), config_check::MAX_DIAGNOSTICS);
    // one line more is refused before any check
    let over = format!("{text}A=1\n");
    let r = check(&mut env, "env", &over).await;
    assert_eq!(r["ok"], json!(false));
    assert!(
        r["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("at most 4000 lines"),
        "{r}"
    );
    // expansions that keep doubling: refused, as the hub's reader would ignore the file
    let mut bomb = format!("A0={}\n", "x".repeat(1024));
    for i in 1..40 {
        bomb.push_str(&format!("A{i}=${{A{}}}${{A{}}}\n", i - 1, i - 1));
    }
    let r = check(&mut env, "env", &bomb).await;
    assert_eq!(r["ok"], json!(false));
    assert!(
        r["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("expansions add up to more than"),
        "{r}"
    );
}

#[tokio::test]
async fn test_env_range_warnings_and_non_finite_floats() {
    let mut env = TestEnv::new();
    let text = "HF_TOKEN_FILE=~/.config/engine-console/hf_token\nBODY_CAP_BYTES=-1\nHOSTS_FILE=/srv/ec/hosts.json\n";
    let r = check(&mut env, "env", text).await;
    assert_eq!(r["ok"], json!(true), "{r}");
    let w: Vec<(&str, u64)> = r["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| {
            (
                w["field"].as_str().unwrap_or(""),
                w["line"].as_u64().unwrap_or(0),
            )
        })
        .collect();
    assert_eq!(w, vec![("hf_token_file", 1), ("body_cap_bytes", 2)], "{r}");
    assert!(!r.to_string().contains("hf_token\""), "{r}");
    for v in ["inf", "nan", "1e999"] {
        let r = check(&mut env, "env", &format!("SSE_IDLE_TIMEOUT_S={v}\n")).await;
        assert_eq!(r["ok"], json!(false), "{v}");
        assert_eq!(r["errors"][0]["field"], json!("sse_idle_timeout_s"));
        assert_eq!(r["errors"][0]["line"], json!(1));
    }
}

#[tokio::test]
async fn test_policy_echoes_of_the_draft_are_cut_short() {
    let mut env = TestEnv::new();
    let long = format!("/data/{}:x", "a".repeat(400));
    let text = json!({"version": 1, "model_roots": [{"id": "r", "path": long}]}).to_string();
    // the node's own message quotes the whole path; the hub's answer keeps 80 characters of it
    let node = ec_hub::services::policy_check::check_policy_text(&text).unwrap_err();
    assert!(node.contains(&long));
    let r = check(&mut env, "policy", &text).await;
    let msg = r["errors"][0]["message"].as_str().unwrap();
    assert!(!msg.contains(&long), "{msg}");
    assert!(
        msg.contains(&format!("'/data/{}…'", "a".repeat(74))),
        "{msg}"
    );
    assert!(msg.ends_with("'..' segments"), "{msg}");
}

#[tokio::test]
async fn test_a_hosts_draft_of_unclosed_quotes_is_answered_at_once() {
    // second audit F3: unknown keys that each start with an apostrophe after a space open a quote that never closes;
    // the message shortener scanned to the end from every one of them (O(k·n) on a 64 KiB draft)
    let mut env = TestEnv::new();
    let mut keys = serde_json::Map::new();
    keys.insert("id".into(), json!("a"));
    // each key is `" 'NNNN":0,` (11 bytes): just under the 64 KiB cap
    for i in 0..(hosts::MAX_FILE_BYTES - 64) / 11 {
        keys.insert(format!(" '{i:04x}"), json!(0));
    }
    let text = json!({"hosts": [keys]}).to_string();
    assert!(
        text.len() <= hosts::MAX_FILE_BYTES && text.len() > hosts::MAX_FILE_BYTES - 128,
        "{}",
        text.len()
    );
    let t0 = std::time::Instant::now();
    let r = check(&mut env, "hosts", &text).await;
    let took = t0.elapsed();
    assert!(took < std::time::Duration::from_millis(500), "{took:?}");
    assert_eq!(r["ok"], json!(false), "{r}");
    let msg = r["errors"][0]["message"].as_str().unwrap();
    assert!(msg.starts_with("hosts[0]: unknown key(s)"), "{msg}");
    assert_eq!(msg.chars().count(), config_check::MAX_MESSAGE_CHARS + 1);
}
