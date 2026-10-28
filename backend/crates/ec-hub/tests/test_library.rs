//! Port of backend/tests/data/test_library.py: fetching and paging, the stored snapshot of a host that is down,
//! grouping, the live overlay (in use, deletable), find(), rescan and delete.

#![allow(clippy::unwrap_used, clippy::float_cmp)]

#[path = "common/wsc_data.rs"]
mod wsc_data;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ec_hub::domain::errors::AppError;
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::library::copy_id;
use ec_hub::services::store::Store;
use serde_json::{Value, json};
use sha1::Digest as _;
use wsc_data::*;

fn node_with(tmp: &Path, host: &str, o: NodeOpts) -> Arc<DataNode> {
    let mut o = o;
    if o.roots.is_empty() {
        o.roots = vec![
            RootCfg::new("hf", "hf-home"),
            RootCfg::new("gguf", "gguf"),
            RootCfg::new("ro", "plain").ro(),
        ];
    }
    o.scan_disk = false;
    node(tmp, host, o)
}

fn n(tmp: &Path, host: &str) -> Arc<DataNode> {
    node_with(tmp, host, NodeOpts::default())
}

fn hf(repo: &str) -> ModelRef {
    ModelRef::hf(repo)
}

fn mref(format: &str, root: &str, rel: &str) -> ModelRef {
    ModelRef {
        format: format.into(),
        repo_id: None,
        root_id: Some(root.into()),
        rel: Some(rel.into()),
        name: None,
    }
}

fn ollama(name: &str) -> ModelRef {
    ModelRef {
        format: "ollama".into(),
        repo_id: None,
        root_id: None,
        rel: None,
        name: Some(name.into()),
    }
}

fn pe(e: &AppError) -> (u16, String, String) {
    (
        e.status(),
        e.code().to_string(),
        e.problem().map(|p| p.detail.clone()).unwrap_or_default(),
    )
}

fn names(w: &World) -> Vec<String> {
    w.library
        .models(None, None, None)
        .unwrap()
        .into_iter()
        .map(|m| m.name)
        .collect()
}

fn list_calls(a: &DataNode) -> Vec<Value> {
    a.ops("models.list")
        .into_iter()
        .map(Value::Object)
        .collect()
}

async fn start(tmp: &Path, nodes: Vec<Arc<DataNode>>, o: WorldOpts) -> World {
    World::new(tmp, nodes, o).start().await
}

#[tokio::test]
async fn test_fetch_pages_through_models_list_until_the_cursor_ends() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = (0..5)
        .map(|i| hf_entry("hf", &format!("org/m{i}"), json!({})))
        .collect();
    let w = start(
        t.path(),
        vec![a.clone()],
        WorldOpts {
            page_limit: Some(2),
            ..WorldOpts::default()
        },
    )
    .await;
    assert_eq!(
        list_calls(&a),
        vec![
            json!({"limit": 2}),
            json!({"limit": 2, "cursor": 2}),
            json!({"limit": 2, "cursor": 4})
        ]
    );
    assert_eq!(
        names(&w),
        (0..5).map(|i| format!("org/m{i}")).collect::<Vec<_>>()
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_scan_finishing_between_two_pages_restarts_the_pass() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = (0..3)
        .map(|i| hf_entry("hf", &format!("org/m{i}"), json!({})))
        .collect();
    a.st().list_generation_bump_once = true;
    let w = start(
        t.path(),
        vec![a.clone()],
        WorldOpts {
            page_limit: Some(2),
            ..WorldOpts::default()
        },
    )
    .await;
    let lists = list_calls(&a);
    assert!(
        lists.len() == 4 && lists[2] == json!({"limit": 2}),
        "{lists:?}"
    );
    assert_eq!(w.library.models(None, None, None).unwrap().len(), 3);
    w.stop().await;
}

#[tokio::test]
async fn test_a_cursor_that_does_not_advance_is_not_followed_forever() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m0", json!({}))];
    a.on(
        "models.list",
        Arc::new(|n, args| {
            let mut out = n.op_models_list(args)?;
            out.insert("next_cursor".into(), json!(1));
            Ok(out)
        }),
    );
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    assert_eq!(list_calls(&a).len(), 2);
    assert_eq!(names(&w), ["org/m0"]);
    w.stop().await;
}

#[tokio::test]
async fn test_copy_id_is_the_first_16_hex_of_sha1_of_host_root_rel() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = start(t.path(), vec![a], WorldOpts::default()).await;
    let rel = "hub/models--org--m";
    let want =
        hex::encode(sha1::Sha1::digest(format!("alpha|hf|{rel}").as_bytes()))[..16].to_string();
    let got = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    assert!(got == want && want == copy_id("alpha", "hf", rel));
    w.stop().await;
}

#[tokio::test]
async fn test_copy_fields_path_drive_and_policy_flags() {
    let t = tempfile::tempdir().unwrap();
    let a = node_with(
        t.path(),
        "alpha",
        NodeOpts {
            label: Some("Alpha box".into()),
            ..NodeOpts::default()
        },
    );
    a.st().entries = vec![hf_entry(
        "hf",
        "org/m",
        json!({"size": 1234, "files": 7, "revision": COMMIT}),
    )];
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    let c = w.library.models(None, None, None).unwrap()[0].copies[0].clone();
    let root = a.root_path("hf");
    assert!(c.host_id == "alpha" && c.host_label == "Alpha box" && c.root_id == "hf");
    assert_eq!(c.root_label.as_deref(), Some("hf"));
    assert_eq!(c.path, format!("{}/hub/models--org--m", root.display()));
    assert!(c.size_bytes == 1234 && c.files == Some(7));
    assert!(
        c.mtime == Some(10.0)
            && c.atime == Some(20.0)
            && c.writable
            && c.deletable
            && c.in_use_by.is_empty()
    );
    assert_eq!(c.visible_on, ["alpha"]);
    let d = c.drive.unwrap();
    assert!(d.mount == "/" && d.label == "/ (ext4, Test SSD)");
    w.stop().await;
}

#[tokio::test]
async fn test_grouping_keys_per_format_and_engines() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    a.st().entries = vec![
        hf_entry("hf", "org/m1", json!({})),
        gguf_entry(
            "gguf",
            "a/Qwen3-8B-Q4_K_M.gguf",
            json!({"name": "Qwen3-8B", "quant": "Q4_K_M"}),
        ),
        gguf_entry(
            "gguf",
            "a/Qwen3-8B-Q8_0.gguf",
            json!({"name": "Qwen3-8B", "quant": "Q8_0"}),
        ),
        entry("ro", "llama-dir", "plain", Some("llama-dir"), json!({})),
        entry("ro", "manifests/x", "ollama", Some("llama3:8b"), json!({})),
        entry("ro", "stuff", "other", Some("stuff"), json!({})),
        entry("ro", "weird", "bogus-format", Some("x"), json!({})),
        entry(
            "gguf",
            "loose/Phi-4-mini.Q6_K.gguf",
            "gguf",
            None,
            json!({"quant": null}),
        ),
    ];
    b.st().entries = vec![hf_entry("hf", "org/m1", json!({"size": 999}))];
    let w = start(t.path(), vec![a, b], WorldOpts::default()).await;
    let models = w.library.models(None, None, None).unwrap();
    let by = |k: &str| models.iter().find(|m| m.key == k).unwrap().clone();
    let mut keys: Vec<String> = models.iter().map(|m| m.key.clone()).collect();
    keys.sort();
    let mut want = vec![
        "hf:org/m1",
        "gguf:Qwen3-8B:Q4_K_M",
        "gguf:Qwen3-8B:Q8_0",
        "plain:llama-dir",
        "ollama:llama3:8b",
        "other:stuff",
        "other:x",
        "gguf:Phi-4-mini.Q6_K:Q6_K",
    ];
    want.sort_unstable();
    assert_eq!(keys, want);
    assert_eq!(
        by("hf:org/m1")
            .copies
            .iter()
            .map(|c| c.host_id.clone())
            .collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    assert_eq!(by("hf:org/m1").size_bytes, 1000);
    assert!(
        by("gguf:Qwen3-8B:Q4_K_M").quant.as_deref() == Some("Q4_K_M")
            && by("gguf:Qwen3-8B:Q4_K_M").format == "gguf"
    );
    assert_eq!(by("hf:org/m1").engines, ["vllm"]);
    assert_eq!(by("gguf:Qwen3-8B:Q8_0").engines, ["vllm", "llamacpp"]);
    assert_eq!(by("plain:llama-dir").engines, ["llamacpp"]);
    assert_eq!(by("ollama:llama3:8b").engines, ["ollama"]);
    assert!(by("other:stuff").engines.is_empty() && by("other:x").format == "other");
    w.stop().await;
}

#[tokio::test]
async fn test_a_quant_already_in_the_name_is_not_added_twice() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![gguf_entry(
        "gguf",
        "x/m.gguf",
        json!({"name": "m:Q4_K_M", "quant": "Q4_K_M"}),
    )];
    let w = start(t.path(), vec![a], WorldOpts::default()).await;
    let keys: Vec<String> = w
        .library
        .models(None, None, None)
        .unwrap()
        .into_iter()
        .map(|m| m.key)
        .collect();
    assert_eq!(keys, ["gguf:m:Q4_K_M"]);
    w.stop().await;
}

#[tokio::test]
async fn test_model_fields_come_from_the_first_copy_that_has_them() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry(
        "hf",
        "org/m",
        json!({"params": 8_000_000_000_i64, "arch": "Qwen3ForCausalLM", "context": 40960, "shards": 4}),
    )];
    let w = start(t.path(), vec![a, b], WorldOpts::default()).await;
    let m = w.library.models(None, None, None).unwrap()[0].clone();
    assert_eq!(
        (
            m.params,
            m.arch.as_deref(),
            m.context,
            m.shards,
            m.revision.as_deref()
        ),
        (
            Some(8_000_000_000),
            Some("Qwen3ForCausalLM"),
            Some(40960),
            Some(4),
            Some(COMMIT)
        )
    );
    w.stop().await;
}

#[tokio::test]
async fn test_copies_at_different_revisions_are_flagged() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    a.st().entries = vec![hf_entry("hf", "org/m", json!({"revision": "a".repeat(40)}))];
    b.st().entries = vec![hf_entry("hf", "org/m", json!({"revision": "b".repeat(40)}))];
    let w = start(t.path(), vec![a, b], WorldOpts::default()).await;
    let ws = w.library.models(None, None, None).unwrap()[0]
        .warnings
        .clone();
    assert!(
        ws.contains(&"copies are at different revisions: aaaaaaaaaaaa, bbbbbbbbbbbb".to_string()),
        "{ws:?}"
    );
    w.stop().await;
}

#[tokio::test]
async fn test_entries_that_break_the_nodes_own_rules_are_not_listed() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![
        hf_entry("hf", "org/ok", json!({})),
        entry("hf", "../etc", "plain", Some("evil"), json!({})),
        entry("nope", "x", "plain", Some("ghost"), json!({})),
        entry("hf", "/abs", "plain", Some("abs"), json!({})),
        entry("hf", "a//b", "plain", Some("empty-seg"), json!({})),
    ];
    let w = start(t.path(), vec![a], WorldOpts::default()).await;
    assert_eq!(names(&w), ["org/ok"]);
    w.stop().await;
}

#[tokio::test]
async fn test_incomplete_scans_and_node_warnings_surface_on_the_model() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry(
        "hf",
        "org/m",
        json!({"warnings": ["no config.json"]}),
    )];
    a.on(
        "models.list",
        Arc::new(|n, args| {
            let mut out = n.op_models_list(args)?;
            out.insert(
                "roots".into(),
                json!([{"root": "hf", "scanned_at": 1.0, "complete": false, "entries": 1, "error": "too many entries"}]),
            );
            Ok(out)
        }),
    );
    let w = start(t.path(), vec![a], WorldOpts::default()).await;
    let ws = w.library.models(None, None, None).unwrap()[0]
        .warnings
        .clone();
    assert!(ws.contains(&"no config.json".to_string()));
    assert!(
        ws.iter()
            .any(|x| x.contains("did not finish: too many entries")),
        "{ws:?}"
    );
    w.stop().await;
}

// ---- stored snapshots ------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_a_host_that_is_down_shows_its_last_known_library_stale() {
    let t = tempfile::tempdir().unwrap();
    let db = t.path().join("console.db");
    let a = n(t.path(), "alpha");
    a.st().entries = vec![
        hf_entry("hf", "org/m", json!({})),
        gguf_entry("gguf", "x/q.gguf", json!({})),
    ];
    let w1 = start(
        t.path(),
        vec![a],
        WorldOpts {
            store: Some(Arc::new(Store::open(&db).unwrap())),
            ..WorldOpts::default()
        },
    )
    .await;
    let mut first: Vec<String> = w1
        .library
        .models(None, None, None)
        .unwrap()
        .into_iter()
        .map(|m| m.key)
        .collect();
    first.sort();
    w1.stop().await;

    let a2 = n(t.path(), "alpha");
    a2.st().state = "down".into();
    a2.forget();
    let w2 = start(
        t.path(),
        vec![a2.clone()],
        WorldOpts {
            store: Some(Arc::new(Store::open(&db).unwrap())),
            ..WorldOpts::default()
        },
    )
    .await;
    let models = w2.library.models(None, None, None).unwrap();
    let mut keys: Vec<String> = models.iter().map(|m| m.key.clone()).collect();
    keys.sort();
    assert_eq!(keys, first);
    for m in &models {
        assert!(
            m.warnings
                .iter()
                .any(|x| x.contains("Alpha is down") && x.contains("last-known")),
            "{:?}",
            m.warnings
        );
        for c in &m.copies {
            assert!(!c.deletable && c.host_id == "alpha");
            assert!(c.path.ends_with("models--org--m") || c.path.ends_with("x/q.gguf"));
            assert_eq!(c.drive.as_ref().unwrap().mount, "/");
        }
    }
    assert_eq!(w2.library.status()["alpha"]["stale"], true);
    assert!(list_calls(&a2).is_empty());
    let e = w2
        .library
        .delete_copy(&models[0].copies[0].id)
        .await
        .unwrap_err();
    assert_eq!(e.status(), 503);
    w2.stop().await;
}

#[tokio::test]
async fn test_stored_snapshots_of_hosts_that_are_no_longer_configured_are_dropped() {
    let t = tempfile::tempdir().unwrap();
    let db = t.path().join("console.db");
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    a.st().entries = vec![hf_entry("hf", "org/a", json!({}))];
    b.st().entries = vec![hf_entry("hf", "org/b", json!({}))];
    let w1 = start(
        t.path(),
        vec![a, b],
        WorldOpts {
            store: Some(Arc::new(Store::open(&db).unwrap())),
            ..WorldOpts::default()
        },
    )
    .await;
    w1.stop().await;
    let w2 = start(
        t.path(),
        vec![n(t.path(), "alpha")],
        WorldOpts {
            store: Some(Arc::new(Store::open(&db).unwrap())),
            ..WorldOpts::default()
        },
    )
    .await;
    assert_eq!(w2.count("SELECT COUNT(*) AS n FROM library_snapshots"), 1);
    w2.stop().await;
}

#[tokio::test]
async fn test_an_unreadable_stored_snapshot_is_dropped_not_fatal() {
    let t = tempfile::tempdir().unwrap();
    let store = Arc::new(Store::memory().unwrap());
    store
        .execute(
            "INSERT INTO library_snapshots(host_id,generation,roots_json,entries_json,updated_at) \
             VALUES('alpha',1,'[]','{not json',1)",
            &[],
        )
        .unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = start(
        t.path(),
        vec![a],
        WorldOpts {
            store: Some(store),
            ..WorldOpts::default()
        },
    )
    .await;
    assert_eq!(names(&w), ["org/m"]);
    w.stop().await;
}

#[tokio::test]
async fn test_the_snapshot_is_replaced_on_every_fetch() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    a.set_entries(vec![
        hf_entry("hf", "org/m", json!({})),
        hf_entry("hf", "org/n", json!({})),
    ]);
    w.library.refresh("alpha").await.unwrap();
    let row = w
        .store
        .one(
            "SELECT generation, entries_json FROM library_snapshots WHERE host_id='alpha'",
            &[],
        )
        .unwrap()
        .unwrap();
    assert_eq!(row.i64("generation"), Some(a.generation()));
    assert_eq!(
        row.text("entries_json")
            .unwrap()
            .matches("\"org/n\"")
            .count(),
        1
    );
    w.stop().await;
}

// ---- following the nodes ----------------------------------------------------------------------------------------
#[tokio::test]
async fn test_models_changed_triggers_a_refetch() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    a.set_entries(vec![
        hf_entry("hf", "org/m", json!({})),
        hf_entry("hf", "org/new", json!({})),
    ]);
    a.announce_models_changed();
    tokio::time::sleep(Duration::from_millis(10)).await;
    w.settle().await;
    let mut got = names(&w);
    got.sort();
    assert_eq!(got, ["org/m", "org/new"]);
    w.stop().await;
}

#[tokio::test]
async fn test_a_burst_of_models_changed_events_collapses_into_few_fetches() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    let before = list_calls(&a).len();
    for _ in 0..20 {
        a.announce_models_changed();
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    w.settle().await;
    assert!(list_calls(&a).len() - before <= 3);
    w.stop().await;
}

#[tokio::test]
async fn test_a_host_coming_up_is_fetched_and_a_host_that_is_down_at_start_is_not() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    let b = node_with(
        t.path(),
        "beta",
        NodeOpts {
            state: "down".into(),
            ..NodeOpts::default()
        },
    );
    a.st().entries = vec![hf_entry("hf", "org/a", json!({}))];
    b.st().entries = vec![hf_entry("hf", "org/b", json!({}))];
    let w = start(t.path(), vec![a, b.clone()], WorldOpts::default()).await;
    assert_eq!(names(&w), ["org/a"]);
    assert!(list_calls(&b).is_empty());
    w.fleet.set_state("beta", "up");
    tokio::time::sleep(Duration::from_millis(10)).await;
    w.settle().await;
    let mut got = names(&w);
    got.sort();
    assert_eq!(got, ["org/a", "org/b"]);
    w.stop().await;
}

#[tokio::test]
async fn test_a_failed_fetch_keeps_the_last_snapshot_and_retries_while_the_host_is_up() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    let fails = Arc::new(std::sync::atomic::AtomicUsize::new(2));
    let f2 = Arc::clone(&fails);
    a.on(
        "models.list",
        Arc::new(move |n, args| {
            if f2.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                f2.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                return Err(ec_hub::domain::errors::NodeError::new(
                    "alpha",
                    "io_error",
                    "the scan is broken",
                ));
            }
            n.op_models_list(args)
        }),
    );
    a.set_entries(vec![
        hf_entry("hf", "org/m", json!({})),
        hf_entry("hf", "org/n", json!({})),
    ]);
    a.announce_models_changed();
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert_eq!(names(&w), ["org/m"]);
    assert!(!w.library.status()["alpha"]["error"].is_null());
    tokio::time::sleep(Duration::from_millis(100)).await;
    w.settle().await;
    let mut got = names(&w);
    got.sort();
    assert_eq!(got, ["org/m", "org/n"]);
    assert!(w.library.status()["alpha"]["error"].is_null());
    w.stop().await;
}

#[tokio::test]
async fn test_a_library_event_is_published_when_a_snapshot_changes() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = World::new(t.path(), vec![a], WorldOpts::default());
    let mut sub = w.bus.subscribe("library");
    let w = w.start().await;
    let ev = tokio::time::timeout(Duration::from_secs(2), sub.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(ev["host_id"] == "alpha" && ev["models"] == 1, "{ev}");
    w.stop().await;
}

// ---- the live overlay ----------------------------------------------------------------------------------------
#[tokio::test]
async fn test_in_use_by_comes_from_the_callback_and_blocks_delete() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![
        hf_entry("hf", "org/m", json!({})),
        hf_entry("hf", "org/free", json!({})),
    ];
    let w = start(t.path(), vec![a], WorldOpts::default()).await;
    let path = w
        .library
        .find_copy("alpha", &hf("org/m"), false)
        .unwrap()
        .path;
    w.set_busy("alpha", &path, &["inst_1"]);
    let c = w.library.find_copy("alpha", &hf("org/m"), false).unwrap();
    assert!(c.in_use_by == ["inst_1"] && !c.deletable);
    assert!(
        w.library
            .find_copy("alpha", &hf("org/free"), false)
            .unwrap()
            .deletable
    );
    w.clear_busy();
    assert!(
        w.library
            .find_copy("alpha", &hf("org/m"), false)
            .unwrap()
            .deletable
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_broken_in_use_callback_does_not_blank_the_library() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let boom: InUse = Arc::new(|_h: &str, _p: &str| -> Vec<String> { panic!("bug") });
    let w = start(
        t.path(),
        vec![a],
        WorldOpts {
            in_use: Some(boom),
            ..WorldOpts::default()
        },
    )
    .await;
    assert_eq!(names(&w), ["org/m"]);
    w.stop().await;
}

#[tokio::test]
async fn test_deletable_needs_policy_a_writable_root_a_live_link_and_no_user() {
    for (mode, allow_delete, writable, state, deletable) in [
        ("manage", true, true, "up", true),
        ("manage", false, true, "up", false),
        ("monitor", true, true, "up", false),
        ("manage", true, false, "up", false),
        ("manage", true, true, "down", false),
    ] {
        let t = tempfile::tempdir().unwrap();
        let mut root = RootCfg::new("hf", "hf-home");
        root.writable = writable;
        let a = node_with(
            t.path(),
            "alpha",
            NodeOpts {
                mode: mode.into(),
                allow_delete,
                roots: vec![root],
                ..NodeOpts::default()
            },
        );
        a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
        let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
        a.st().state = state.into();
        let c = w.library.models(None, None, None).unwrap()[0].copies[0].clone();
        assert!(
            c.deletable == deletable && c.writable == writable,
            "{mode} {allow_delete} {writable} {state}"
        );
        w.stop().await;
    }
}

#[tokio::test]
async fn test_a_policy_change_shows_without_a_new_scan() {
    let t = tempfile::tempdir().unwrap();
    let a = node_with(
        t.path(),
        "alpha",
        NodeOpts {
            allow_delete: false,
            ..NodeOpts::default()
        },
    );
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    assert!(!w.library.models(None, None, None).unwrap()[0].copies[0].deletable);
    a.st().allow_delete = true;
    assert!(w.library.models(None, None, None).unwrap()[0].copies[0].deletable);
    w.stop().await;
}

// ---- find ---------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_find_by_format() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    a.st().entries = vec![
        hf_entry("hf", "org/m", json!({})),
        gguf_entry("gguf", "x/q-Q4_K_M.gguf", json!({})),
        entry("ro", "pdir", "plain", Some("pdir"), json!({})),
        entry("ro", "manifests/l", "ollama", Some("llama3:8b"), json!({})),
    ];
    b.st().entries = vec![
        hf_entry("hf", "org/m", json!({})),
        gguf_entry("gguf", "x/q-Q4_K_M.gguf", json!({})),
    ];
    let w = start(t.path(), vec![a, b], WorldOpts::default()).await;
    let lib = &w.library;
    assert_eq!(
        lib.find_copy("alpha", &hf("org/m"), false).unwrap().host_id,
        "alpha"
    );
    assert_eq!(
        lib.find_copy("beta", &hf("org/m"), false).unwrap().host_id,
        "beta"
    );
    assert!(lib.find_copy("alpha", &hf("org/none"), false).is_none());
    let g = lib
        .find_copy("beta", &mref("gguf", "gguf", "x/q-Q4_K_M.gguf"), false)
        .unwrap();
    assert!(g.host_id == "beta" && g.root_id == "gguf");
    assert!(
        lib.find_copy("alpha", &mref("gguf", "gguf", "x/other.gguf"), false)
            .is_none()
    );
    assert!(
        lib.find_copy("alpha", &mref("plain", "ro", "pdir"), false)
            .is_some()
    );
    assert!(
        lib.find_copy("alpha", &mref("gguf", "ro", "pdir"), false)
            .is_none()
    );
    assert!(
        lib.find_copy("alpha", &ollama("llama3:8b"), false)
            .is_some()
    );
    assert!(lib.find_copy("beta", &ollama("llama3:8b"), false).is_none());
    w.stop().await;
}

#[tokio::test]
async fn test_an_hf_directory_with_no_snapshot_is_listed_but_is_not_a_model_an_engine_can_load() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    a.st().entries = vec![hf_entry(
        "hf",
        "org/m",
        json!({"revision": null, "files": 0, "size": 500, "warnings": ["no snapshot: the download has not finished"]}),
    )];
    b.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = start(t.path(), vec![a, b], WorldOpts::default()).await;
    let m = w.library.models(None, None, None).unwrap()[0].clone();
    assert_eq!(
        m.copies
            .iter()
            .map(|c| c.host_id.clone())
            .collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    assert!(
        m.warnings
            .contains(&"no snapshot: the download has not finished".to_string())
    );
    assert!(w.library.find_copy("alpha", &hf("org/m"), false).is_none());
    assert_eq!(
        w.library
            .find_copy("beta", &hf("org/m"), false)
            .unwrap()
            .host_id,
        "beta"
    );
    assert_eq!(
        w.library
            .copies_of("alpha")
            .into_iter()
            .map(|c| c.rel)
            .collect::<Vec<_>>(),
        ["hub/models--org--m"]
    );
    w.stop().await;
}

#[tokio::test]
async fn test_copies_lists_what_is_physically_on_a_host() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    a.st().entries = vec![
        hf_entry("hf", "org/a1", json!({})),
        hf_entry("hf", "org/a2", json!({})),
    ];
    b.st().entries = vec![hf_entry("hf", "org/b1", json!({}))];
    let w = start(t.path(), vec![a, b], WorldOpts::default()).await;
    assert_eq!(
        w.library
            .copies_of("alpha")
            .into_iter()
            .map(|c| c.host_id)
            .collect::<Vec<_>>(),
        ["alpha", "alpha"]
    );
    assert_eq!(w.library.copies_of("beta").len(), 1);
    assert!(w.library.copies_of("nobody").is_empty());
    assert!(w.library.copy(&"0".repeat(16)).is_none() && w.library.copy("not-an-id").is_none());
    w.stop().await;
}

// ---- the list API -----------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_models_filters_by_host_format_and_query_words() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    a.st().entries = vec![
        hf_entry("hf", "Qwen/Qwen3-8B", json!({})),
        gguf_entry("gguf", "x/Llama-3-Q4_K_M.gguf", json!({"name": "Llama-3"})),
    ];
    b.st().entries = vec![
        hf_entry("hf", "Qwen/Qwen3-8B", json!({})),
        hf_entry("hf", "org/Gemma-3", json!({})),
    ];
    let w = start(t.path(), vec![a, b], WorldOpts::default()).await;
    let lib = &w.library;
    let nm = |v: Vec<ec_hub::domain::fleet::LibraryModel>| {
        v.into_iter().map(|m| m.name).collect::<Vec<_>>()
    };
    assert_eq!(
        nm(lib.models(Some("beta"), None, None).unwrap()),
        ["org/Gemma-3", "Qwen/Qwen3-8B"]
    );
    assert_eq!(
        nm(lib.models(None, Some("gguf"), None).unwrap()),
        ["Llama-3"]
    );
    assert_eq!(
        nm(lib.models(None, None, Some("qwen 8b")).unwrap()),
        ["Qwen/Qwen3-8B"]
    );
    assert_eq!(
        nm(lib.models(None, None, Some("Q4_K_M")).unwrap()),
        ["Llama-3"]
    );
    assert!(
        lib.models(None, None, Some("nothing like this"))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        lib.models(Some("beta"), None, Some("qwen")).unwrap()[0]
            .copies
            .iter()
            .map(|c| c.host_id.clone())
            .collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    assert!(lib.model("hf:Qwen/Qwen3-8B").is_some() && lib.model("hf:nope").is_none());
    w.stop().await;
}

#[tokio::test]
async fn test_local_models_is_the_v1_shape_for_every_host() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    a.st().entries = vec![
        hf_entry("hf", "org/m", json!({"size": 10})),
        gguf_entry("gguf", "x/q.gguf", json!({})),
    ];
    b.st().entries = vec![hf_entry(
        "hf",
        "org/m",
        json!({"size": 20, "revision": "b".repeat(40)}),
    )];
    let w = start(t.path(), vec![a, b], WorldOpts::default()).await;
    let rows = w.library.local_models().unwrap();
    let got: Vec<(String, String, i64, Vec<String>)> = rows
        .iter()
        .map(|m| {
            (
                m.repo_id.clone(),
                m.host_id.clone(),
                m.size_bytes,
                m.revisions.clone(),
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            ("org/m".into(), "alpha".into(), 10, vec![COMMIT.to_string()]),
            ("org/m".into(), "beta".into(), 20, vec!["b".repeat(40)])
        ]
    );
    assert!(rows[0].last_used == Some(20.0) && rows[0].path.ends_with("hub/models--org--m"));
    w.stop().await;
}

// ---- commands -------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_rescan_asks_the_connected_nodes_and_refetches() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    let c = node_with(
        t.path(),
        "gamma",
        NodeOpts {
            state: "down".into(),
            ..NodeOpts::default()
        },
    );
    let w = start(
        t.path(),
        vec![a.clone(), b.clone(), c.clone()],
        WorldOpts::default(),
    )
    .await;
    assert_eq!(w.library.rescan(None).await.unwrap(), ["alpha", "beta"]);
    let force = vec![serde_json::Map::from_iter([(
        "force".to_string(),
        json!(true),
    )])];
    assert!(a.scans() == force && b.scans() == force && c.scans().is_empty());
    assert_eq!(w.library.rescan(Some("beta")).await.unwrap(), ["beta"]);
    let e = w.library.rescan(Some("nobody")).await.unwrap_err();
    assert_eq!(e.code(), "unknown_host");
    let e = w.library.rescan(Some("gamma")).await.unwrap_err();
    let (s, code, detail) = pe(&e);
    assert!(
        s == 503 && code == "node_unavailable" && detail.starts_with("Gamma: "),
        "{detail}"
    );
    w.stop().await;
}

#[tokio::test]
async fn test_rescan_of_all_hosts_skips_one_that_refuses() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (n(t.path(), "alpha"), n(t.path(), "beta"));
    let w = start(t.path(), vec![a.clone(), b], WorldOpts::default()).await;
    a.on(
        "models.scan",
        Arc::new(|_n, _a| {
            Err(ec_hub::domain::errors::NodeError::new(
                "alpha",
                "busy",
                "scanning already",
            ))
        }),
    );
    assert_eq!(w.library.rescan(None).await.unwrap(), ["beta"]);
    let e = w.library.rescan(Some("alpha")).await.unwrap_err();
    assert!(e.status() == 503 && e.code() == "busy");
    w.stop().await;
}

#[tokio::test]
async fn test_delete_removes_the_files_and_the_entry_at_once() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    let repo = a.root_path("hf").join("hub").join("models--org--m");
    std::fs::create_dir_all(repo.join("blobs")).unwrap();
    std::fs::write(repo.join("blobs").join("x"), b"12345").unwrap();
    a.st().entries = vec![
        hf_entry("hf", "org/m", json!({})),
        hf_entry("hf", "org/keep", json!({})),
    ];
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    let cid = w.library.models(None, None, Some("org/m")).unwrap()[0].copies[0]
        .id
        .clone();
    w.library.delete_copy(&cid).await.unwrap();
    assert_eq!(
        a.ops("fs.delete")
            .into_iter()
            .map(Value::Object)
            .collect::<Vec<_>>(),
        [json!({"root": "hf", "rel": "hub/models--org--m"})]
    );
    assert!(!repo.exists());
    assert_eq!(names(&w), ["org/keep"]);
    assert_eq!(a.scans().len(), 1);
    let ej = w
        .store
        .one("SELECT entries_json FROM library_snapshots", &[])
        .unwrap()
        .unwrap();
    assert_eq!(ej.text("entries_json").unwrap().matches("org/m").count(), 0);
    a.announce_models_changed();
    tokio::time::sleep(Duration::from_millis(10)).await;
    w.settle().await;
    assert_eq!(names(&w), ["org/keep"]);
    w.stop().await;
}

#[tokio::test]
async fn test_delete_refusals() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![
        hf_entry("hf", "org/m", json!({})),
        entry("ro", "pdir", "plain", Some("pdir"), json!({})),
        entry("hf", ".", "plain", Some("the-whole-root"), json!({})),
    ];
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    let lib = &w.library;
    let cid = lib.find_copy("alpha", &hf("org/m"), false).unwrap().id;
    let e0 = lib.delete_copy(&"f".repeat(16)).await.unwrap_err();
    assert_eq!(e0.code(), "copy_not_found");
    assert_eq!(
        lib.delete_copy("not an id").await.unwrap_err().status(),
        404
    );
    let p = lib.find_copy("alpha", &hf("org/m"), false).unwrap().path;
    w.set_busy("alpha", &p, &["inst_9"]);
    let (s, code, detail) = pe(&lib.delete_copy(&cid).await.unwrap_err());
    assert!(s == 409 && code == "model_in_use" && detail.contains("inst_9"));
    w.clear_busy();
    let ro = lib
        .find_copy("alpha", &mref("plain", "ro", "pdir"), false)
        .unwrap()
        .id;
    let (s, code, detail) = pe(&lib.delete_copy(&ro).await.unwrap_err());
    assert!(s == 403 && code == "policy_denied" && detail.starts_with("Alpha: "));
    let root_entry = lib
        .copies_of("alpha")
        .into_iter()
        .find(|c| c.rel.is_empty())
        .unwrap();
    assert_eq!(
        lib.delete_copy(&root_entry.id).await.unwrap_err().code(),
        "root_not_deletable"
    );
    a.st().allow_delete = false;
    let e4 = lib.delete_copy(&cid).await.unwrap_err();
    assert!(e4.status() == 403 && e4.code() == "policy_denied");
    a.st().allow_delete = true;
    a.st().mode = "monitor".into();
    let e5 = lib.delete_copy(&cid).await.unwrap_err();
    assert!(e5.status() == 409 && e5.code() == "monitor_only");
    a.st().mode = "manage".into();
    a.st().in_use_paths.insert(p.clone());
    let e6 = lib.delete_copy(&cid).await.unwrap_err();
    assert!(e6.status() == 409 && e6.code() == "in_use");
    assert_eq!(
        lib.models(None, None, Some("org/m"))
            .unwrap()
            .into_iter()
            .map(|m| m.name)
            .collect::<Vec<_>>(),
        ["org/m"]
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_stale_host_cannot_delete() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    a.st().state = "connecting".into();
    let e = w.library.delete_copy(&cid).await.unwrap_err();
    assert!(e.status() == 503 && e.code() == "node_unavailable");
    assert!(a.ops("fs.delete").is_empty());
    w.stop().await;
}

#[tokio::test]
async fn test_start_is_idempotent_and_stop_ends_the_subscriptions() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path(), "alpha");
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    let w = start(t.path(), vec![a.clone()], WorldOpts::default()).await;
    w.library.start().await;
    assert_eq!(list_calls(&a).len(), 1);
    w.library.stop().await;
    a.set_entries(Vec::new());
    a.announce_models_changed();
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(names(&w), ["org/m"]);
    w.stop().await;
}
