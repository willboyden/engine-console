//! Port of backend/tests/data/test_settings_pins.py: image pins per host (host pin -> engine pin -> none) and the
//! shape rules of PUT /settings `host_image_pins`, against the SettingsService (the Python file
//! is a data test). Python's `image_pin(engine)` defaults the host to "local"; Rust passes it.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::sync::Arc;

use ec_hub::config::Settings;
use ec_hub::domain::errors::AppError;
use ec_hub::domain::models::SettingsPatch;
use ec_hub::services::settings::{SettingsService, clean_host_image_pins};
use ec_hub::services::store::Store;
use serde_json::{Map, Value, json};

const DIGEST_HEX: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn make_settings(tmp: &Path) -> Settings {
    let mut s = Settings::defaults();
    s.data_dir = tmp.join("data");
    s.hf_cache_dir = tmp.join("hf");
    s.otlp_enabled = false;
    s.background_tasks = false;
    s.hf_token = None;
    s.hf_token_file = tmp.join("no-token");
    s.fleet_enabled = false;
    s
}

fn svc(tmp: &Path) -> (SettingsService, Arc<Store>) {
    let store = Arc::new(Store::memory().unwrap());
    let s = SettingsService::new(
        Arc::clone(&store),
        Arc::new(make_settings(tmp)),
        Box::new(|| vec!["fake/engine:2".to_string(), "fake/engine:1".to_string()]),
    );
    (s, store)
}

fn put(s: &SettingsService, patch: Value) -> Result<Value, AppError> {
    let p: SettingsPatch = serde_json::from_value(patch).unwrap();
    s.update(&p)
}

fn pins(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        other => panic!("not a table: {other}"),
    }
}

fn code(e: &AppError) -> String {
    match e {
        AppError::Problem(p) => {
            assert_eq!(p.status, 400, "{p:?}");
            p.code.clone()
        }
        AppError::Node(n) => panic!("{n:?}"),
    }
}

fn pin(s: &SettingsService, engine: &str, host: &str) -> Option<String> {
    s.image_pin(engine, host).unwrap()
}

#[test]
fn test_no_pins_by_default() {
    let t = tempfile::tempdir().unwrap();
    let (s, _) = svc(t.path());
    assert!(s.host_image_pins().unwrap().is_empty());
    assert!(pin(&s, "vllm", "local").is_none() && pin(&s, "vllm", "spark").is_none());
    assert_eq!(s.effective().unwrap()["host_image_pins"], json!({}));
}

#[test]
fn test_a_host_pin_beats_the_engine_pin_which_beats_nothing() {
    let t = tempfile::tempdir().unwrap();
    let (s, _) = svc(t.path());
    put(&s, json!({"image_pins": {"vllm": "fake/engine:2"}})).unwrap();
    // the global pin applies
    assert_eq!(pin(&s, "vllm", "local").as_deref(), Some("fake/engine:2"));
    assert_eq!(pin(&s, "vllm", "spark").as_deref(), Some("fake/engine:2"));
    put(
        &s,
        json!({"host_image_pins": {"spark": {"vllm": "vllm/vllm-openai:v0.12.0-arm64"}}}),
    )
    .unwrap();
    assert_eq!(
        pin(&s, "vllm", "spark").as_deref(),
        Some("vllm/vllm-openai:v0.12.0-arm64")
    );
    assert_eq!(pin(&s, "vllm", "local").as_deref(), Some("fake/engine:2"));
    // no pin for another engine
    assert!(pin(&s, "sglang", "spark").is_none());
    let eff = s.effective().unwrap();
    assert_eq!(
        eff["host_image_pins"],
        json!({"spark": {"vllm": "vllm/vllm-openai:v0.12.0-arm64"}})
    );
    // untouched by host pins
    assert_eq!(eff["image_pins"], json!({"vllm": "fake/engine:2"}));
}

#[test]
fn test_local_is_the_default_host() {
    let t = tempfile::tempdir().unwrap();
    let (s, _) = svc(t.path());
    put(&s, json!({"host_image_pins": {"local": {"vllm": "a/b:1"}}})).unwrap();
    assert_eq!(pin(&s, "vllm", "local").as_deref(), Some("a/b:1"));
}

#[test]
fn test_a_put_replaces_the_whole_table_and_empty_clears_it() {
    let t = tempfile::tempdir().unwrap();
    let (s, _) = svc(t.path());
    put(
        &s,
        json!({"host_image_pins": {"spark": {"vllm": "a/b:1"}, "gpu2": {"sglang": "c/d:2"}}}),
    )
    .unwrap();
    put(
        &s,
        json!({"host_image_pins": {"gpu2": {"sglang": "c/d:3"}}}),
    )
    .unwrap();
    assert_eq!(
        s.host_image_pins().unwrap(),
        pins(json!({"gpu2": {"sglang": "c/d:3"}}))
    );
    // a patch without the field leaves it alone
    put(&s, json!({"idle_ttl_s": 10})).unwrap();
    assert_eq!(
        s.host_image_pins().unwrap(),
        pins(json!({"gpu2": {"sglang": "c/d:3"}}))
    );
    put(&s, json!({"host_image_pins": {}})).unwrap();
    assert!(s.host_image_pins().unwrap().is_empty());
}

#[test]
fn test_a_host_with_no_pins_is_dropped() {
    let t = tempfile::tempdir().unwrap();
    let (s, _) = svc(t.path());
    put(
        &s,
        json!({"host_image_pins": {"spark": {}, "gpu2": {"vllm": "a/b:1"}}}),
    )
    .unwrap();
    assert_eq!(
        s.host_image_pins().unwrap(),
        pins(json!({"gpu2": {"vllm": "a/b:1"}}))
    );
}

/// Whether a machine may run an image is that machine's node policy (an arm64-only image is not in the hub's list).
/// The global pin keeps the v1 rule.
#[test]
fn test_host_pins_are_not_held_to_the_hubs_allowlist() {
    let t = tempfile::tempdir().unwrap();
    let (s, _) = svc(t.path());
    put(
        &s,
        json!({"host_image_pins": {"spark": {"vllm": "registry.example/team/vllm-arm64:1.2.3"}}}),
    )
    .unwrap();
    assert_eq!(
        pin(&s, "vllm", "spark").as_deref(),
        Some("registry.example/team/vllm-arm64:1.2.3")
    );
    let e = put(
        &s,
        json!({"image_pins": {"vllm": "registry.example/team/vllm-arm64:1.2.3"}}),
    )
    .unwrap_err();
    assert_eq!(code(&e), "image_not_allowed");
    let e2 = put(&s, json!({"image_pins": {"vllm": "fake/engine:latest"}})).unwrap_err();
    assert_eq!(code(&e2), "invalid_setting");
}

/// The six parametrized images of `test_pinned_tags_and_digests_are_accepted`.
#[test]
fn test_pinned_tags_and_digests_are_accepted() {
    let long = format!("{}:1", "a".repeat(200));
    let digest = format!("repo/name@sha256:{DIGEST_HEX}");
    for image in [
        "a/b:1",
        "vllm/vllm-openai:v0.12.0",
        digest.as_str(),
        "ghcr.io/ggml-org/llama.cpp:server-cuda-b6000",
        "registry.example:5000/team/img:1.0+build.5",
        long.as_str(),
    ] {
        let table = pins(json!({"spark": {"vllm": image}}));
        assert_eq!(clean_host_image_pins(&table).unwrap(), table, "{image}");
    }
}

/// The thirteen parametrized images of `test_images_that_are_not_pinned_references_are_refused`.
#[test]
fn test_images_that_are_not_pinned_references_are_refused() {
    let long_tag = format!("a/b:{}", "x".repeat(300));
    for image in [
        "a/b",
        "a/b:latest",
        "latest",
        "",
        "a/b:1 --privileged",
        "a/b:1\n--rm",
        "-a/b:1",
        "a/b:1;rm -rf /",
        "a b:1",
        "$(x):1",
        long_tag.as_str(),
        "é/b:1",
        "a/b:1\0",
    ] {
        let e = clean_host_image_pins(&pins(json!({"spark": {"vllm": image}}))).unwrap_err();
        assert_eq!(
            (e.status, e.code.as_str()),
            (400, "invalid_setting"),
            "{image:?}"
        );
    }
}

/// The ten parametrized hosts of `test_host_ids_follow_the_host_id_rule`.
#[test]
fn test_host_ids_follow_the_host_id_rule() {
    let long = "a".repeat(33);
    for host in [
        "Spark",
        "-x",
        "a b",
        long.as_str(),
        "",
        "a/b",
        "a\n",
        "ü",
        "UPPER",
        "a_b",
    ] {
        let e = clean_host_image_pins(&pins(json!({host: {"vllm": "a/b:1"}}))).unwrap_err();
        assert_eq!(
            (e.status, e.code.as_str()),
            (400, "invalid_setting"),
            "{host:?}"
        );
    }
    let ok = pins(
        json!({"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {"vllm": "a/b:1"}, "gpu-2": {"vllm": "a/b:1"},
                         "0x": {"vllm": "a/b:1"}}),
    );
    assert!(!clean_host_image_pins(&ok).unwrap().is_empty());
}

/// The seven parametrized engines of `test_engine_ids_are_checked_too`.
#[test]
fn test_engine_ids_are_checked_too() {
    let long = "a".repeat(65);
    for engine in ["", "has space", "-lead", long.as_str(), "a/b", "a\n", "é"] {
        let e = clean_host_image_pins(&pins(json!({"spark": {engine: "a/b:1"}}))).unwrap_err();
        assert_eq!(e.status, 400, "{engine:?}");
    }
    let ok = pins(json!({"spark": {"llama.cpp": "a/b:1", "trt_llm-2": "a/b:1", "V1": "a/b:1"}}));
    assert!(!clean_host_image_pins(&ok).unwrap().is_empty());
}

#[test]
fn test_the_table_is_bounded() {
    let hosts = |n: usize| {
        let mut m = Map::new();
        for i in 0..n {
            m.insert(format!("h{i}"), json!({"vllm": "a/b:1"}));
        }
        m
    };
    assert!(clean_host_image_pins(&hosts(33)).is_err());
    let mut engines = Map::new();
    for i in 0..17 {
        engines.insert(format!("e{i}"), json!("a/b:1"));
    }
    assert!(clean_host_image_pins(&pins(json!({"spark": engines}))).is_err());
    assert_eq!(clean_host_image_pins(&hosts(32)).unwrap().len(), 32);
}

#[test]
fn test_a_refused_change_changes_nothing() {
    let t = tempfile::tempdir().unwrap();
    let (s, _) = svc(t.path());
    put(&s, json!({"host_image_pins": {"spark": {"vllm": "a/b:1"}}})).unwrap();
    assert!(
        put(
            &s,
            json!({"idle_ttl_s": 99, "host_image_pins": {"spark": {"vllm": "a/b:latest"}}})
        )
        .is_err()
    );
    assert_eq!(
        s.host_image_pins().unwrap(),
        pins(json!({"spark": {"vllm": "a/b:1"}}))
    );
    assert_eq!(s.idle_ttl_s().unwrap(), 0);
}

#[test]
fn test_a_damaged_stored_value_reads_as_no_pins() {
    let t = tempfile::tempdir().unwrap();
    let (s, store) = svc(t.path());
    for raw in [
        r#""just a string""#,
        "[1, 2]",
        r#"{"spark": "x"}"#,
        r#"{"spark": {"vllm": 5}}"#,
        r#"{"spark": {"vllm": "a/b:1"}, "x": 1}"#,
    ] {
        store
            .execute(
                "INSERT INTO settings(key,value) VALUES('host_image_pins',?) ON CONFLICT(key) DO UPDATE SET \
                 value=excluded.value",
                &[&raw],
            )
            .unwrap();
        for (h, v) in s.host_image_pins().unwrap() {
            let v = v.as_object().unwrap_or_else(|| panic!("{raw}: {h}"));
            assert!(v.values().all(Value::is_string), "{raw}");
        }
    }
    assert_eq!(pin(&s, "vllm", "spark").as_deref(), Some("a/b:1"));
    store
        .execute(
            "UPDATE settings SET value='\"x\"' WHERE key='host_image_pins'",
            &[],
        )
        .unwrap();
    assert!(s.host_image_pins().unwrap().is_empty() && pin(&s, "vllm", "spark").is_none());
}

#[test]
fn test_v1_methods_are_all_still_there() {
    let t = tempfile::tempdir().unwrap();
    let (s, _) = svc(t.path());
    // the method set is the type's (a missing one would not compile); their v1 behaviour:
    assert!(s.engine_image_allowed("fake/engine:2"));
    assert_eq!(s.hub_dir(), s.hf_cache_dir().join("hub"));
    assert_eq!(s.idle_ttl_s().unwrap(), 0);
    assert!(s.default_gpu_ids().unwrap().is_empty());
    assert_eq!(s.token_status(), "[unset]");
    assert!(s.hf_token().is_none());
    assert!(s.effective().unwrap().is_object());
    let key = s.create_key("k", "admin").unwrap();
    let secret = key.secret.clone().unwrap();
    assert!(s.authenticate(&secret).unwrap().is_some());
    assert!(s.list_keys().unwrap()[0].secret.is_none());
    s.revoke_key(&key.id).unwrap();
    assert!(s.authenticate(&secret).unwrap().is_none());
    let boot = t.path().join("boot");
    std::fs::create_dir_all(&boot).unwrap();
    let _ = s.ensure_bootstrap_key(&boot).unwrap();
}
