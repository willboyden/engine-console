//! Port of backend/tests/data/test_downloads.py: downloads into any machine's model roots — layouts, resume,
//! verification, refusals, pause/resume/cancel, progress.
//!
//! Python checks the written cache with huggingface_hub's `scan_cache_dir`; here `hf_cache_repos` reads the same
//! structure (models--org--name / snapshots/<commit>/<files as relative links into blobs/> / refs/<name>) and the
//! byte- and link-level assertions are the same as Python's.

#![allow(clippy::unwrap_used, clippy::float_cmp)]

#[path = "common/wsc_data.rs"]
mod wsc_data;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use ec_hub::domain::errors::{AppError, egress_proxy_unavailable};
use ec_hub::domain::models::{Download, DownloadRequest};
use ec_hub::domain::ports::RepoFile;
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::downloads::{blob_name, files_dir, hf_repo_rel, layout_of, snapshot_link};
use ec_hub::services::library::{RootSpec, RootUsage};
use ec_hub::services::store::Store;
use serde_json::{Value, json};
use wsc_data::*;

fn spark(tmp: &Path, o: NodeOpts) -> Arc<DataNode> {
    let mut o = o;
    if o.roots.is_empty() {
        o.roots = vec![
            RootCfg::new("hf", "hf-home"),
            RootCfg::new("hub", "hf-hub"),
            RootCfg::new("gguf", "gguf"),
            RootCfg::new("ro", "hf-home").ro(),
            RootCfg::new("pl", "plain"),
            RootCfg::new("ol", "ollama"),
        ];
    }
    if o.addresses.is_empty() {
        o.addresses = vec!["10.0.0.5".into()];
    }
    o.label = o.label.or_else(|| Some("Spark".into()));
    node(tmp, "spark", o)
}

async fn make(tmp: &Path, nd: Option<Arc<DataNode>>, o: WorldOpts) -> (World, Arc<DataNode>) {
    let nd = nd.unwrap_or_else(|| spark(tmp, NodeOpts::default()));
    let w = World::new(tmp, vec![Arc::clone(&nd)], o);
    w.hub
        .add_files("org/m", &files(), Some(json!({"model_type": "x"})));
    (w.start().await, nd)
}

fn req(repo: &str, host: &str, root: Option<&str>) -> DownloadRequest {
    DownloadRequest {
        repo_id: repo.into(),
        revision: None,
        allow_patterns: None,
        host_id: host.into(),
        root_id: root.map(str::to_string),
    }
}

async fn run_req(w: &World, r: DownloadRequest) -> Download {
    let d = w.downloads.create(&r).await.unwrap();
    w.downloads.wait(&d.id).await;
    w.downloads.get(&d.id).unwrap()
}

async fn run(w: &World, root: &str) -> Download {
    run_req(w, req("org/m", "spark", Some(root))).await
}

fn rel_w() -> String {
    format!("hub/models--org--m/blobs/{}", sha())
}

fn writes_of(nd: &DataNode, rel: &str) -> Vec<Value> {
    nd.writes()
        .into_iter()
        .filter(|x| x["rel"] == rel)
        .collect()
}

/// huggingface_hub's scan_cache_dir, reduced to what the tests compare: (repo id, [commits], files per revision).
fn hf_cache_repos(dir: &Path) -> Vec<(String, Vec<String>, usize)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    let mut dirs: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    dirs.sort();
    for d in dirs {
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let Some(rest) = name.strip_prefix("models--") else {
            continue;
        };
        let repo = rest.replacen("--", "/", 1);
        let mut commits = Vec::new();
        let mut nfiles = 0;
        for s in std::fs::read_dir(d.join("snapshots")).unwrap().flatten() {
            commits.push(s.file_name().to_string_lossy().to_string());
            let mut stack = vec![s.path()];
            while let Some(p) = stack.pop() {
                for f in std::fs::read_dir(&p).unwrap().flatten() {
                    let fp = f.path();
                    let meta = fp.symlink_metadata().unwrap();
                    if meta.file_type().is_symlink() {
                        let target = std::fs::read_link(&fp).unwrap();
                        assert!(target.is_relative(), "{}", fp.display());
                        let resolved = std::fs::canonicalize(&fp).unwrap();
                        assert!(
                            resolved.starts_with(std::fs::canonicalize(d.join("blobs")).unwrap())
                        );
                        nfiles += 1;
                    } else if meta.is_dir() {
                        stack.push(fp);
                    } else {
                        nfiles += 1;
                    }
                }
            }
        }
        for r in std::fs::read_dir(d.join("refs")).unwrap().flatten() {
            let c = std::fs::read_to_string(r.path()).unwrap();
            assert!(commits.contains(&c), "a ref to a missing snapshot");
        }
        out.push((repo, commits, nfiles));
    }
    out
}

fn tick() -> ec_hub::services::downloads::Clock {
    let t = Arc::new(AtomicU64::new(0f64.to_bits()));
    Arc::new(move || {
        let v = f64::from_bits(t.load(Ordering::SeqCst)) + 0.3;
        t.store(v.to_bits(), Ordering::SeqCst);
        v
    })
}

async fn drain(sub: &mut ec_hub::services::common::Subscription) -> Vec<Value> {
    let mut out = Vec::new();
    while let Ok(Some(ev)) = tokio::time::timeout(Duration::from_millis(20), sub.recv()).await {
        out.push(ev);
    }
    out
}

fn pe(e: &AppError) -> (u16, String, String) {
    (
        e.status(),
        e.code().to_string(),
        e.problem().map(|p| p.detail.clone()).unwrap_or_default(),
    )
}

// ---- layouts (pure) -----------------------------------------------------------------------------------------------
#[test]
fn test_layout_helpers() {
    assert_eq!(hf_repo_rel("hf-home", "org/m"), "hub/models--org--m");
    assert_eq!(hf_repo_rel("hf-hub", "org/m"), "models--org--m");
    assert_eq!(files_dir("org/m"), "org--m");
    assert_eq!(
        snapshot_link("hub/models--o--m", COMMIT, "w.bin", "abc"),
        (
            format!("hub/models--o--m/snapshots/{COMMIT}/w.bin"),
            "../../blobs/abc".to_string()
        )
    );
    assert_eq!(
        snapshot_link("models--o--m", COMMIT, "onnx/m.onnx", "abc").1,
        "../../../blobs/abc"
    );
    let rf = |p: &str, size: i64, sha: Option<&str>| RepoFile {
        path: p.into(),
        size,
        sha256: sha.map(str::to_string),
    };
    assert_eq!(
        blob_name(&rf("w", 1, Some(&"f".repeat(64))), COMMIT),
        "f".repeat(64)
    );
    let small = blob_name(&rf("config.json", 2, None), COMMIT);
    assert!(small.starts_with("nolfs-") && small.len() == 46);
    assert_ne!(
        blob_name(&rf("config.json", 2, None), &"b".repeat(40)),
        small
    );
    assert_ne!(blob_name(&rf("other.json", 2, None), COMMIT), small);
}

fn usage(
    entries: i64,
    hf_home: bool,
    hf_hub: bool,
    loose: bool,
    ollama: bool,
    other: bool,
) -> RootUsage {
    RootUsage {
        entries,
        hf_home,
        hf_hub,
        loose,
        ollama,
        other,
    }
}

#[test]
fn test_layout_of_a_root() {
    let e = RootUsage::default();
    for (kind, u, layout) in [
        ("hf-home", e, Some("hf-home")),
        ("hf-hub", e, Some("hf-hub")),
        ("gguf", e, Some("files")),
        ("plain", e, Some("files")),
        ("ollama", e, None),
        ("other", e, None),
        ("auto", e, Some("hf-home")),
        (
            "auto",
            usage(1, true, false, false, false, false),
            Some("hf-home"),
        ),
        (
            "auto",
            usage(1, false, true, false, false, false),
            Some("hf-hub"),
        ),
        (
            "auto",
            usage(2, false, false, true, false, false),
            Some("files"),
        ),
        ("auto", usage(2, false, false, true, false, true), None),
        ("auto", usage(1, false, false, false, true, false), None),
        ("auto", usage(1, false, false, false, false, true), None),
    ] {
        let spec = RootSpec {
            id: "r".into(),
            path: "/x".into(),
            kind: kind.into(),
            writable: true,
            label: None,
        };
        assert_eq!(layout_of(&spec, &u, "hf-home"), layout, "{kind} {u:?}");
    }
}

#[test]
fn test_an_empty_auto_root_takes_what_the_caller_is_writing() {
    let auto = RootSpec {
        id: "r".into(),
        path: "/x".into(),
        kind: "auto".into(),
        writable: true,
        label: None,
    };
    assert_eq!(
        layout_of(&auto, &RootUsage::default(), "hf-home"),
        Some("hf-home")
    );
    assert_eq!(
        layout_of(&auto, &RootUsage::default(), "files"),
        Some("files")
    );
    assert_eq!(
        layout_of(&auto, &usage(1, false, true, false, false, false), "files"),
        Some("hf-hub")
    );
}

// ---- hf-home, hf-hub and gguf roots ------------------------------------------------------------------------------
#[tokio::test]
async fn test_download_into_an_hf_home_root_writes_a_real_hf_cache() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    let d = run(&w, "hf").await;
    assert!(
        d.state == "completed"
            && d.host_id == "spark"
            && d.root_id.as_deref() == Some("hf")
            && d.kind == "hf"
    );
    assert_eq!(
        (
            d.total_bytes,
            d.done_bytes,
            d.files_total,
            d.files_done,
            d.commit_sha.as_deref()
        ),
        (TOTAL, TOTAL, 3, 3, Some(COMMIT))
    );
    assert!(d.error.is_none() && d.current_file.is_none());
    let repo = nd.root_path("hf").join("hub").join("models--org--m");
    assert_eq!(
        std::fs::read(repo.join("blobs").join(sha())).unwrap(),
        PAYLOAD
    );
    let link = repo.join("snapshots").join(COMMIT).join("w.safetensors");
    assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
    assert_eq!(
        std::fs::read_link(&link).unwrap().to_string_lossy(),
        format!("../../blobs/{}", sha())
    );
    assert_eq!(std::fs::read(&link).unwrap(), PAYLOAD);
    assert_eq!(
        std::fs::read(repo.join("snapshots").join(COMMIT).join("config.json")).unwrap(),
        b"{}"
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("refs").join("main")).unwrap(),
        COMMIT
    );
    let mut inc = Vec::new();
    collect(&repo, &mut inc, ".incomplete");
    assert!(inc.is_empty());
    let repos = hf_cache_repos(&nd.root_path("hf").join("hub"));
    assert_eq!(
        repos,
        vec![("org/m".to_string(), vec![COMMIT.to_string()], 3)]
    );
    w.settle().await;
    assert_eq!(nd.scans(), vec![obj(json!({"force": true}))]);
    let found = w
        .library
        .find_copy("spark", &ModelRef::hf("org/m"), false)
        .unwrap();
    assert!(found.size_bytes == TOTAL && found.root_id == "hf");
    w.stop().await;
}

fn collect(dir: &Path, out: &mut Vec<std::path::PathBuf>, suffix: &str) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.symlink_metadata().unwrap().is_dir() {
                collect(&p, out, suffix);
            } else if p.to_string_lossy().ends_with(suffix) {
                out.push(p);
            }
        }
    }
}

#[tokio::test]
async fn test_download_into_an_hf_hub_root_has_no_hub_directory() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    let d = run(&w, "hub").await;
    assert_eq!(d.state, "completed");
    let root = nd.root_path("hub");
    assert!(
        root.join("models--org--m")
            .join("blobs")
            .join(sha())
            .is_file()
    );
    assert!(
        root.join("models--org--m")
            .join("refs")
            .join("main")
            .is_file()
    );
    assert!(!root.join("hub").exists());
    assert_eq!(
        hf_cache_repos(&root)
            .into_iter()
            .map(|r| r.0)
            .collect::<Vec<_>>(),
        ["org/m"]
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_revision_names_the_ref() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    let mut r = req("org/m", "spark", Some("hf"));
    r.revision = Some("v1.0".into());
    let d = run_req(&w, r).await;
    assert!(d.state == "completed" && d.revision.as_deref() == Some("v1.0"));
    let refs = nd.root_path("hf").join("hub/models--org--m/refs/v1.0");
    assert_eq!(std::fs::read_to_string(refs).unwrap(), COMMIT);
    w.stop().await;
}

#[tokio::test]
async fn test_a_gguf_root_gets_the_selected_files_under_the_repo_name() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    let (q4, q8) = (b"q4".repeat(20), b"q8".repeat(30));
    w.hub.add(
        "unsloth/Qwen3-GGUF",
        &[
            ("Qwen3-Q4_K_M.gguf", &q4),
            ("Qwen3-Q8_0.gguf", &q8),
            ("README.md", b"hi"),
            ("sub/extra.gguf", b"zz"),
        ],
    );
    let mut r = req("unsloth/Qwen3-GGUF", "spark", Some("gguf"));
    r.allow_patterns = Some(vec!["Qwen3-Q4_K_M.gguf".into()]);
    let d = run_req(&w, r).await;
    assert!(d.state == "completed" && d.total_bytes == q4.len() as i64 && d.files_total == 1);
    let target = nd.root_path("gguf").join("unsloth--Qwen3-GGUF");
    let mut all = Vec::new();
    collect(&target, &mut all, "");
    assert_eq!(
        all.iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>(),
        ["Qwen3-Q4_K_M.gguf"]
    );
    assert_eq!(std::fs::read(target.join("Qwen3-Q4_K_M.gguf")).unwrap(), q4);
    assert_eq!(nd.writes()[0]["sha256"], sha_hex(&q4));
    assert!(!target.join("blobs").exists());
    w.settle().await;
    let keys: Vec<String> = w
        .library
        .models(None, Some("gguf"), None)
        .unwrap()
        .into_iter()
        .map(|m| m.key)
        .collect();
    assert_eq!(keys, ["gguf:Qwen3-Q4_K_M:Q4_K_M"]);
    w.stop().await;
}

#[tokio::test]
async fn test_a_gguf_root_refuses_a_selection_with_no_gguf_file() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(t.path(), None, WorldOpts::default()).await;
    let d = run(&w, "gguf").await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("no_gguf_files"));
    w.stop().await;
}

#[tokio::test]
async fn test_a_gguf_root_replaces_an_older_file_of_the_same_size() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    let new = vec![b'N'; 40];
    w.hub.add("org/g", &[("m-Q4_K_M.gguf", &new)]);
    let old = nd.root_path("gguf").join("org--g").join("m-Q4_K_M.gguf");
    std::fs::create_dir_all(old.parent().unwrap()).unwrap();
    std::fs::write(&old, vec![b'O'; 40]).unwrap();
    let d = run_req(&w, req("org/g", "spark", Some("gguf"))).await;
    assert!(d.state == "completed" && std::fs::read(&old).unwrap() == new);
    w.stop().await;
}

#[tokio::test]
async fn test_a_small_file_is_a_new_blob_at_a_new_commit() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    run(&w, "hf").await;
    let mut f2 = files();
    f2[0].1 = br#"{"v": 2}"#.to_vec();
    let f2r: Vec<(&str, &[u8])> = f2.iter().map(|(p, b)| (p.as_str(), b.as_slice())).collect();
    w.hub.add_full(
        "org/m",
        &f2r,
        &"d".repeat(40),
        Some(json!({"model_type": "x"})),
    );
    let d = run(&w, "hf").await;
    assert_eq!(d.state, "completed");
    let blobs_dir = nd.root_path("hf").join("hub/models--org--m/blobs");
    let blobs: Vec<String> = std::fs::read_dir(&blobs_dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert_eq!(blobs.iter().filter(|b| b.starts_with("nolfs-")).count(), 4);
    assert_eq!(blobs.iter().filter(|b| **b == sha()).count(), 1);
    let cfg = nd.root_path("hf").join(format!(
        "hub/models--org--m/snapshots/{}/config.json",
        "d".repeat(40)
    ));
    assert_eq!(std::fs::read(cfg).unwrap(), br#"{"v": 2}"#);
    w.stop().await;
}

#[tokio::test]
async fn test_two_files_with_the_same_content_share_a_blob() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    w.hub.add(
        "org/twin",
        &[("a.safetensors", PAYLOAD), ("b.safetensors", PAYLOAD)],
    );
    let d = run_req(&w, req("org/twin", "spark", Some("hf"))).await;
    assert!(
        d.state == "completed" && d.done_bytes == 100 && d.total_bytes == 100 && d.files_done == 2
    );
    let snaps = nd.root_path("hf").join("hub/models--org--twin");
    assert_eq!(std::fs::read_dir(snaps.join("blobs")).unwrap().count(), 1);
    assert_eq!(
        std::fs::read(snaps.join(format!("snapshots/{COMMIT}/a.safetensors"))).unwrap(),
        std::fs::read(snaps.join(format!("snapshots/{COMMIT}/b.safetensors"))).unwrap()
    );
    assert_eq!(
        nd.writes()
            .iter()
            .filter(|x| x["rel"].as_str().unwrap().ends_with(&sha()))
            .count(),
        1
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_zero_byte_file_needs_no_download_request() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    w.hub
        .add("org/e", &[("empty.bin", b""), ("w.safetensors", PAYLOAD)]);
    let d = run_req(&w, req("org/e", "spark", Some("hf"))).await;
    assert!(d.state == "completed" && d.total_bytes == 50);
    assert_eq!(
        w.hub.opens().into_iter().map(|o| o.2).collect::<Vec<_>>(),
        ["w.safetensors"]
    );
    let e = nd
        .root_path("hf")
        .join(format!("hub/models--org--e/snapshots/{COMMIT}/empty.bin"));
    assert_eq!(std::fs::read(e).unwrap(), b"");
    w.stop().await;
}

#[tokio::test]
async fn test_allow_patterns_select_files() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(t.path(), None, WorldOpts::default()).await;
    let mut r = req("org/m", "spark", Some("hf"));
    r.allow_patterns = Some(vec!["*.json".into()]);
    assert_eq!(run_req(&w, r).await.files_total, 1);
    let mut r2 = req("org/m", "spark", Some("hub"));
    r2.allow_patterns = Some(vec!["nothing*".into()]);
    let d = run_req(&w, r2).await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("no_files"));
    w.stop().await;
}

// ---- default roots and auto roots ---------------------------------------------------------------------------------
#[tokio::test]
async fn test_the_default_root_is_the_first_writable_hf_root() {
    let t = tempfile::tempdir().unwrap();
    let nd = spark(
        t.path(),
        NodeOpts {
            roots: vec![
                RootCfg::new("ro", "hf-home").ro(),
                RootCfg::new("g", "gguf"),
                RootCfg::new("pl", "plain"),
                RootCfg::new("hub", "hf-hub"),
                RootCfg::new("hf", "hf-home"),
            ],
            ..NodeOpts::default()
        },
    );
    let (w, _) = make(t.path(), Some(nd), WorldOpts::default()).await;
    let d = w
        .downloads
        .create(&req("org/m", "spark", None))
        .await
        .unwrap();
    assert_eq!(d.root_id.as_deref(), Some("hub"));
    w.downloads.wait(&d.id).await;
    assert_eq!(w.downloads.get(&d.id).unwrap().state, "completed");
    w.stop().await;
}

#[tokio::test]
async fn test_an_empty_auto_root_starts_an_hf_cache_and_one_with_models_keeps_its_kind() {
    let t = tempfile::tempdir().unwrap();
    let nd = spark(
        t.path(),
        NodeOpts {
            roots: vec![
                RootCfg::new("a1", "auto"),
                RootCfg::new("a2", "auto"),
                RootCfg::new("a3", "auto"),
                RootCfg::new("a4", "auto"),
            ],
            ..NodeOpts::default()
        },
    );
    let w = World::new(t.path(), vec![Arc::clone(&nd)], WorldOpts::default());
    w.hub.add_files("org/m", &files(), None);
    {
        let mut st = nd.st();
        st.scan_disk_enabled = false;
        st.entries = vec![
            entry("a2", "models--x--y", "hf", Some("x/y"), json!({})),
            gguf_entry("a3", "loose.gguf", json!({})),
            entry("a4", "manifests/q", "ollama", Some("q:1"), json!({})),
        ];
    }
    let w = w.start().await;
    let d1 = run(&w, "a1").await;
    assert!(d1.state == "completed" && nd.root_path("a1").join("hub/models--org--m").is_dir());
    let d2 = run(&w, "a2").await;
    assert!(d2.state == "completed" && nd.root_path("a2").join("models--org--m").is_dir());
    let mut r3 = req("org/m", "spark", Some("a3"));
    r3.allow_patterns = Some(vec!["config.json".into()]);
    let d3 = run_req(&w, r3).await;
    assert!(d3.state == "completed" && nd.root_path("a3").join("org--m/config.json").is_file());
    let e = w
        .downloads
        .create(&req("org/m", "spark", Some("a4")))
        .await
        .unwrap_err();
    assert_eq!(e.code(), "root_kind_unsupported");
    w.stop().await;
}

// ---- resume and verification ------------------------------------------------------------------------------------
fn seed_partial(nd: &DataNode, root: &str, rel: &str, data: &[u8]) -> std::path::PathBuf {
    let part = partial_of(nd, root, rel);
    std::fs::create_dir_all(part.parent().unwrap()).unwrap();
    std::fs::write(&part, data).unwrap();
    part
}

#[tokio::test]
async fn test_a_partial_on_the_node_is_resumed_not_restarted() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    let part = seed_partial(&nd, "hf", &rel_w(), &PAYLOAD[..20]);
    let d = run(&w, "hf").await;
    assert!(d.state == "completed" && d.done_bytes == TOTAL);
    assert!(
        w.hub
            .opens()
            .contains(&("org/m".into(), COMMIT.into(), "w.safetensors".into(), 20))
    );
    assert_eq!(
        writes_of(&nd, &rel_w())[0],
        json!({"rel": rel_w(), "offset": 20, "size": 50, "sha256": sha()})
    );
    assert_eq!(
        std::fs::read(nd.root_path("hf").join(rel_w())).unwrap(),
        PAYLOAD
    );
    assert!(!part.exists());
    w.stop().await;
}

#[tokio::test]
async fn test_a_server_that_ignores_range_restarts_the_file_cleanly() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    w.hub.st().ignore_range = true;
    seed_partial(&nd, "hf", &rel_w(), &PAYLOAD[..20]);
    let d = run(&w, "hf").await;
    assert!(d.state == "completed" && d.done_bytes == TOTAL && d.total_bytes == TOTAL);
    assert_eq!(writes_of(&nd, &rel_w())[0]["offset"], 0);
    assert_eq!(
        std::fs::read(nd.root_path("hf").join(rel_w())).unwrap(),
        PAYLOAD
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_partial_that_is_already_whole_only_needs_the_nodes_verification() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    seed_partial(&nd, "hf", &rel_w(), PAYLOAD);
    let d = run(&w, "hf").await;
    assert_eq!(d.state, "completed");
    assert!(!w.hub.opens().iter().any(|o| o.2 == "w.safetensors"));
    w.stop().await;
}

#[tokio::test]
async fn test_a_partial_longer_than_the_file_starts_over() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    seed_partial(&nd, "hf", &rel_w(), &[b'x'; 500]);
    let d = run(&w, "hf").await;
    assert!(d.state == "completed" && writes_of(&nd, &rel_w())[0]["offset"] == 0);
    w.stop().await;
}

#[tokio::test]
async fn test_a_hash_mismatch_discards_the_partial_and_a_resume_refetches() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    w.hub.st().corrupt.insert("w.safetensors".into());
    let d = run(&w, "hf").await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("hash_mismatch"));
    assert!(d.error.as_deref().unwrap().starts_with("Spark: "));
    assert!(
        !partial_of(&nd, "hf", &rel_w()).exists() && !nd.root_path("hf").join(rel_w()).exists()
    );
    w.hub.st().corrupt.clear();
    w.downloads.resume(&d.id).await.unwrap();
    w.downloads.wait(&d.id).await;
    let d2 = w.downloads.get(&d.id).unwrap();
    assert!(d2.state == "completed" && d2.error.is_none() && d2.error_code.is_none());
    assert_eq!(
        w.hub
            .opens()
            .iter()
            .rfind(|o| o.2 == "w.safetensors")
            .unwrap()
            .3,
        0
    );
    assert_eq!(
        std::fs::read(nd.root_path("hf").join(rel_w())).unwrap(),
        PAYLOAD
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_node_io_error_midway_keeps_the_partial_and_resume_continues_from_it() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    nd.st().fail_after_bytes = Some(30);
    let d = run(&w, "hf").await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("io_error"));
    assert!(d.error.as_deref().unwrap().starts_with("Spark: "));
    assert!(exists_with_size(&partial_of(&nd, "hf", &rel_w()), 30));
    w.downloads.resume(&d.id).await.unwrap();
    w.downloads.wait(&d.id).await;
    assert_eq!(w.downloads.get(&d.id).unwrap().state, "completed");
    assert_eq!(writes_of(&nd, &rel_w()).last().unwrap()["offset"], 30);
    assert!(
        w.hub
            .opens()
            .contains(&("org/m".into(), COMMIT.into(), "w.safetensors".into(), 30))
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_source_that_ends_early_keeps_the_partial_and_reports_a_size_mismatch() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    set_file(&w, "org/m", "w.safetensors", &PAYLOAD[..35]);
    let d = run(&w, "hf").await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("size_mismatch"));
    assert!(exists_with_size(&partial_of(&nd, "hf", &rel_w()), 35));
    set_file(&w, "org/m", "w.safetensors", PAYLOAD);
    w.downloads.resume(&d.id).await.unwrap();
    w.downloads.wait(&d.id).await;
    assert_eq!(w.downloads.get(&d.id).unwrap().state, "completed");
    w.stop().await;
}

fn set_file(w: &World, repo: &str, path: &str, data: &[u8]) {
    let mut st = w.hub.st();
    let files = &mut st.repos.get_mut(repo).unwrap().1;
    files.iter_mut().find(|(p, _)| p == path).unwrap().1 = data.to_vec();
}

#[tokio::test]
async fn test_a_source_that_sends_too_much_is_stopped() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    w.hub.st().repos.get_mut("org/m").unwrap().0.files[1].size = 30;
    let d = run(&w, "hf").await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("size_mismatch"));
    assert!(
        d.error
            .as_deref()
            .unwrap()
            .contains("more than the declared 30")
    );
    let p = partial_of(&nd, "hf", &rel_w());
    assert!(std::fs::metadata(&p).unwrap().len() <= 30);
    w.stop().await;
}

#[tokio::test]
async fn test_a_hostile_hub_listing_fails_the_download_before_anything_is_written() {
    type Mutate = Box<dyn Fn(&mut ec_hub::domain::ports::HubModel)>;
    let add = |p: &'static str, size: i64, sha: Option<&'static str>| -> Mutate {
        Box::new(move |hm| {
            hm.files.push(RepoFile {
                path: p.into(),
                size,
                sha256: sha.map(str::to_string),
            });
        })
    };
    let cases: Vec<(Mutate, &str)> = vec![
        (add("../evil", 1, None), "hf_bad_path"),
        (add("/abs", 1, None), "hf_bad_path"),
        (add("a\\b", 1, None), "hf_bad_path"),
        (add("a/./b", 1, None), "hf_bad_path"),
        (add("x.bin", 1, Some("NOT-A-HASH")), "hf_bad_hash"),
        (add("x.bin", -1, None), "hf_bad_size"),
        (Box::new(|hm| hm.sha = Some("short".into())), "hf_bad_sha"),
        (Box::new(|hm| hm.sha = None), "hf_bad_sha"),
    ];
    for (mutate, code) in cases {
        let t = tempfile::tempdir().unwrap();
        let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
        mutate(&mut w.hub.st().repos.get_mut("org/m").unwrap().0);
        let d = run(&w, "hf").await;
        assert!(
            d.state == "failed" && d.error_code.as_deref() == Some(code),
            "{code}: {:?}",
            d.error_code
        );
        assert!(nd.writes().is_empty());
        assert!(
            !nd.calls()
                .iter()
                .any(|(op, _)| op == "fs.symlink" || op == "fs.put_small")
        );
        w.stop().await;
    }
}

#[tokio::test]
async fn test_a_gated_repo_fails_with_the_hubs_message() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(t.path(), None, WorldOpts::default()).await;
    w.hub.st().gated.insert("org/m".into());
    let d = run(&w, "hf").await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("hf_gated"));
    assert!(d.error.as_deref().unwrap().contains("gated"));
    w.stop().await;
}

#[tokio::test]
async fn test_an_egress_proxy_that_is_down_fails_the_download_closed_and_writes_nothing() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    w.hub.st().model_override = Some(Arc::new(|_| {
        Err(
            egress_proxy_unavailable("egress proxy unreachable: refusing to connect directly")
                .into(),
        )
    }));
    let d = run(&w, "hf").await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("egress_proxy_unavailable"));
    assert!(d.error.as_deref().unwrap().contains("directly"));
    assert!(nd.writes().is_empty() && w.hub.opens().is_empty());
    w.stop().await;
}

#[tokio::test]
async fn test_the_proxy_going_away_midway_keeps_the_partial_for_a_resume() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    w.hub.st().fail_mid = Some((
        "w.safetensors".into(),
        2,
        3,
        egress_proxy_unavailable("egress proxy unreachable: refusing to connect directly"),
    ));
    let d = run(&w, "hf").await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("egress_proxy_unavailable"));
    assert!(exists_with_size(&partial_of(&nd, "hf", &rel_w()), 16));
    w.downloads.resume(&d.id).await.unwrap();
    w.downloads.wait(&d.id).await;
    assert_eq!(w.downloads.get(&d.id).unwrap().state, "completed");
    assert_eq!(
        std::fs::read(nd.root_path("hf").join(rel_w())).unwrap(),
        PAYLOAD
    );
    w.stop().await;
}

// ---- space ----------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_the_disk_precheck_uses_the_free_space_of_the_target_mount() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    nd.st().free = 10;
    let d = run(&w, "hf").await;
    let err = d.error.clone().unwrap_or_default();
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("insufficient_disk"));
    assert!(err.contains("Spark") && err.contains("GB free"), "{err}");
    assert!(nd.writes().is_empty());
    nd.st().free = 1_000_000_000_000;
    w.downloads.resume(&d.id).await.unwrap();
    w.downloads.wait(&d.id).await;
    assert_eq!(w.downloads.get(&d.id).unwrap().state, "completed");
    w.stop().await;
}

#[tokio::test]
async fn test_bytes_already_on_the_node_do_not_count_against_the_free_space() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    run(&w, "hf").await;
    nd.st().free = 5;
    w.store.execute("DELETE FROM downloads", &[]).unwrap();
    assert_eq!(run(&w, "hf").await.state, "completed");
    w.stop().await;
}

#[tokio::test]
async fn test_a_node_that_runs_out_of_space_reads_as_insufficient_disk() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    nd.st().space_left = Some(20);
    let d = run(&w, "hf").await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("insufficient_disk"));
    assert!(d.error.as_deref().unwrap().starts_with("Spark: "));
    w.stop().await;
}

// ---- refusals at create -------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_refusals_say_why_and_leave_no_row() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    let refused = |host: &'static str, root: Option<&'static str>| {
        let w = &w;
        async move {
            pe(&w
                .downloads
                .create(&req("org/m", host, root))
                .await
                .unwrap_err())
        }
    };
    let (s, c, _) = refused("nobody", None).await;
    assert_eq!((s, c.as_str()), (404, "unknown_host"));
    let (s, c, d) = refused("spark", Some("ro")).await;
    assert!(
        s == 409 && c == "root_not_writable" && d.contains("read-only") && d.starts_with("Spark: ")
    );
    let (s, c, _) = refused("spark", Some("nope")).await;
    assert_eq!((s, c.as_str()), (422, "unknown_root"));
    for kr in ["pl", "ol"] {
        let (s, c, _) = refused("spark", Some(kr)).await;
        assert_eq!((s, c.as_str()), (422, "root_kind_unsupported"));
    }
    nd.st().state = "down".into();
    let (s, c, d) = refused("spark", None).await;
    assert!(s == 503 && c == "node_unavailable" && d.starts_with("Spark: "));
    nd.st().state = "up".into();
    nd.st().mode = "monitor".into();
    let (s, c, d) = refused("spark", None).await;
    assert!(s == 409 && c == "monitor_only" && d.contains("monitor"));
    nd.st().mode = "manage".into();
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM downloads"), 0);
    w.stop().await;
}

#[tokio::test]
async fn test_no_writable_root_that_can_hold_a_repo() {
    let t = tempfile::tempdir().unwrap();
    let nd = spark(
        t.path(),
        NodeOpts {
            roots: vec![
                RootCfg::new("ro", "hf-home").ro(),
                RootCfg::new("pl", "plain"),
            ],
            ..NodeOpts::default()
        },
    );
    let (w, _) = make(t.path(), Some(nd), WorldOpts::default()).await;
    let e = w
        .downloads
        .create(&req("org/m", "spark", None))
        .await
        .unwrap_err();
    assert_eq!(e.code(), "no_writable_root");
    w.stop().await;
}

#[tokio::test]
async fn test_bad_requests() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(t.path(), None, WorldOpts::default()).await;
    for bad in ["../etc", "a--b/c", "x"] {
        let e = w
            .downloads
            .create(&req(bad, "spark", None))
            .await
            .unwrap_err();
        assert_eq!(e.code(), "invalid_repo_id");
    }
    for pats in [
        vec!["x".to_string(); 65],
        vec!["a".repeat(300)],
        vec![String::new()],
    ] {
        let mut r = req("org/m", "spark", None);
        r.allow_patterns = Some(pats);
        let e = w.downloads.create(&r).await.unwrap_err();
        assert!(e.status() == 400 && e.code() == "invalid_params");
    }
    w.stop().await;
}

#[tokio::test]
async fn test_a_run_time_refusal_is_a_failure_that_can_be_retried() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    w.hub.st().corrupt.insert("w.safetensors".into());
    let d = run(&w, "hf").await;
    assert_eq!(d.error_code.as_deref(), Some("hash_mismatch"));
    nd.st().mode = "monitor".into();
    w.downloads.resume(&d.id).await.unwrap();
    w.downloads.wait(&d.id).await;
    let d2 = w.downloads.get(&d.id).unwrap();
    assert!(d2.state == "failed" && d2.error_code.as_deref() == Some("monitor_only"));
    nd.st().mode = "manage".into();
    w.hub.st().corrupt.clear();
    w.downloads.resume(&d.id).await.unwrap();
    w.downloads.wait(&d.id).await;
    assert_eq!(w.downloads.get(&d.id).unwrap().state, "completed");
    w.stop().await;
}

// ---- queue ----------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_the_queue_is_capped_and_a_repeat_of_a_running_download_is_refused() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(
        t.path(),
        None,
        WorldOpts {
            max_queue: Some(2),
            ..WorldOpts::default()
        },
    )
    .await;
    w.hub.st().gate = Some(Gate::new());
    w.hub.add_files("org/two", &files(), None);
    w.hub.add_files("org/three", &files(), None);
    let d1 = w
        .downloads
        .create(&req("org/m", "spark", Some("hf")))
        .await
        .unwrap();
    assert_eq!(
        w.downloads
            .create(&req("org/m", "spark", Some("hf")))
            .await
            .unwrap_err()
            .code(),
        "download_exists"
    );
    w.downloads
        .create(&req("org/two", "spark", Some("hf")))
        .await
        .unwrap();
    assert_eq!(
        w.downloads
            .create(&req("org/three", "spark", Some("hf")))
            .await
            .unwrap_err()
            .code(),
        "download_queue_full"
    );
    w.downloads.cancel(&d1.id).await.unwrap();
    w.stop().await;
}

#[tokio::test]
async fn test_the_same_repo_may_go_to_two_different_roots() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(t.path(), None, WorldOpts::default()).await;
    let a = w
        .downloads
        .create(&req("org/m", "spark", Some("hf")))
        .await
        .unwrap();
    let b = w
        .downloads
        .create(&req("org/m", "spark", Some("hub")))
        .await
        .unwrap();
    w.downloads.wait(&a.id).await;
    w.downloads.wait(&b.id).await;
    assert!(
        w.downloads.get(&a.id).unwrap().state == "completed"
            && w.downloads.get(&b.id).unwrap().state == "completed"
    );
    w.stop().await;
}

#[tokio::test]
async fn test_recover_parks_running_rows() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(t.path(), None, WorldOpts::default()).await;
    w.store
        .execute("INSERT INTO downloads(id,repo_id,state,created_at,updated_at) VALUES('d','a/b','running',1,1)", &[])
        .unwrap();
    w.store
        .execute(
            "INSERT INTO downloads(id,repo_id,state,created_at,updated_at,kind,host_id) VALUES('c','a/b','queued',1,1,'copy','x')",
            &[],
        )
        .unwrap();
    w.downloads.recover().unwrap();
    let (d, c) = (w.downloads.get("d").unwrap(), w.downloads.get("c").unwrap());
    assert!(d.state == "paused" && c.state == "paused");
    assert!(d.host_id == "local" && d.kind == "hf" && d.root_id.is_none());
    w.stop().await;
}

#[tokio::test]
async fn test_a_v1_database_keeps_its_downloads_across_the_migration() {
    let t = tempfile::tempdir().unwrap();
    let db = t.path().join("v1.db");
    {
        let con = rusqlite::Connection::open(&db).unwrap();
        con.execute(
            "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, applied_at REAL)",
            [],
        )
        .unwrap();
        for (name, sql) in ec_hub::services::store::MIGRATIONS.iter().take(2) {
            for stmt in ec_hub::services::store::split_sql(sql) {
                con.execute_batch(&stmt).unwrap();
            }
            con.execute("INSERT INTO schema_migrations VALUES (?, 1)", [name])
                .unwrap();
        }
        con.execute(
            "INSERT INTO downloads(id,repo_id,revision,state,total_bytes,done_bytes,created_at,updated_at) \
             VALUES('dl_old','org/old','main','completed',10,10,1,2)",
            [],
        )
        .unwrap();
    }
    let store = Arc::new(Store::open(&db).unwrap());
    let names: Vec<String> = store
        .all("SELECT name FROM schema_migrations", &[])
        .unwrap()
        .iter()
        .map(|r| r.str_or("name", ""))
        .collect();
    assert!(names.contains(&"0004_library_and_transfers.sql".to_string()));
    let w = World::new(
        t.path(),
        vec![spark(t.path(), NodeOpts::default())],
        WorldOpts {
            store: Some(store),
            ..WorldOpts::default()
        },
    );
    let d = w.downloads.get("dl_old").unwrap();
    assert_eq!(
        (
            d.state.as_str(),
            d.repo_id.as_str(),
            d.revision.as_deref(),
            d.host_id.as_str(),
            d.root_id.as_deref(),
            d.kind.as_str(),
            d.source_copy_id.as_deref()
        ),
        (
            "completed",
            "org/old",
            Some("main"),
            "local",
            None,
            "hf",
            None
        )
    );
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM library_snapshots"), 0);
    w.stop().await;
}

#[tokio::test]
async fn test_list_and_get() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(t.path(), None, WorldOpts::default()).await;
    let d = run(&w, "hf").await;
    let page = w.downloads.list_page(10, None).unwrap();
    assert!(
        page.items.iter().map(|x| x.id.clone()).collect::<Vec<_>>() == [d.id.clone()]
            && page.next_cursor.is_none()
    );
    assert_eq!(w.downloads.get("nope").unwrap_err().status(), 404);
    w.stop().await;
}

// ---- pause, resume, cancel ------------------------------------------------------------------------------------------
async fn until_partial(nd: &DataNode, root: &str, rel: &str, size: u64) {
    let p = partial_of(nd, root, rel);
    until(|| exists_with_size(&p, size), "the partial").await;
}

#[tokio::test]
async fn test_pause_keeps_the_partial_and_resume_continues_from_it() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    stop_midway(&w, 3);
    let d = w
        .downloads
        .create(&req("org/m", "spark", Some("hf")))
        .await
        .unwrap();
    until_partial(&nd, "hf", &rel_w(), 32).await;
    let paused = w.downloads.pause(&d.id).await.unwrap();
    assert!(paused.state == "paused" && paused.current_file.as_deref() == Some("w.safetensors"));
    assert!(exists_with_size(&partial_of(&nd, "hf", &rel_w()), 32));
    release(&w);
    w.downloads.resume(&d.id).await.unwrap();
    w.downloads.wait(&d.id).await;
    let done = w.downloads.get(&d.id).unwrap();
    assert!(done.state == "completed" && done.done_bytes == TOTAL);
    assert!(
        w.hub
            .opens()
            .contains(&("org/m".into(), COMMIT.into(), "w.safetensors".into(), 32))
    );
    assert_eq!(writes_of(&nd, &rel_w()).last().unwrap()["offset"], 32);
    assert_eq!(
        std::fs::read(nd.root_path("hf").join(rel_w())).unwrap(),
        PAYLOAD
    );
    w.stop().await;
}

#[tokio::test]
async fn test_cancel_removes_the_partial_when_the_nodes_policy_allows() {
    let t = tempfile::tempdir().unwrap();
    for (allow, left) in [(true, false), (false, true)] {
        let sub = t.path().join(allow.to_string());
        let nd = spark(
            &sub,
            NodeOpts {
                allow_delete: allow,
                ..NodeOpts::default()
            },
        );
        let (w, nd) = make(&sub, Some(nd), WorldOpts::default()).await;
        stop_midway(&w, 3);
        let d = w
            .downloads
            .create(&req("org/m", "spark", Some("hf")))
            .await
            .unwrap();
        until_partial(&nd, "hf", &rel_w(), 32).await;
        assert_eq!(w.downloads.cancel(&d.id).await.unwrap().state, "cancelled");
        assert_eq!(partial_of(&nd, "hf", &rel_w()).exists(), left);
        assert_eq!(w.downloads.cancel(&d.id).await.unwrap_err().status(), 409);
        w.stop().await;
    }
}

#[tokio::test]
async fn test_cancelling_a_paused_download_cleans_up_too() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    stop_midway(&w, 3);
    let d = w
        .downloads
        .create(&req("org/m", "spark", Some("hf")))
        .await
        .unwrap();
    until_partial(&nd, "hf", &rel_w(), 32).await;
    w.downloads.pause(&d.id).await.unwrap();
    assert!(partial_of(&nd, "hf", &rel_w()).exists());
    w.downloads.cancel(&d.id).await.unwrap();
    assert!(
        !partial_of(&nd, "hf", &rel_w()).exists()
            && w.downloads.get(&d.id).unwrap().state == "cancelled"
    );
    w.stop().await;
}

#[tokio::test]
async fn test_cancelling_a_failed_download_removes_the_partial_it_left() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    nd.st().fail_after_bytes = Some(30);
    let d = run(&w, "hf").await;
    assert!(d.state == "failed" && exists_with_size(&partial_of(&nd, "hf", &rel_w()), 30));
    assert_eq!(w.downloads.cancel(&d.id).await.unwrap().state, "cancelled");
    assert!(!partial_of(&nd, "hf", &rel_w()).exists());
    w.stop().await;
}

#[tokio::test]
async fn test_deleting_a_running_download_stops_it_and_removes_the_partial() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    stop_midway(&w, 3);
    let d = w
        .downloads
        .create(&req("org/m", "spark", Some("hf")))
        .await
        .unwrap();
    until_partial(&nd, "hf", &rel_w(), 32).await;
    w.downloads.delete(&d.id).await.unwrap();
    assert!(!partial_of(&nd, "hf", &rel_w()).exists());
    assert_eq!(w.downloads.get(&d.id).unwrap_err().status(), 404);
    w.stop().await;
}

#[tokio::test]
async fn test_pause_and_cancel_right_after_create_are_honoured_before_the_first_step() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(t.path(), None, WorldOpts::default()).await;
    let d = w
        .downloads
        .create(&req("org/m", "spark", Some("hf")))
        .await
        .unwrap();
    assert_eq!(w.downloads.pause(&d.id).await.unwrap().state, "paused");
    assert_eq!(w.downloads.pause(&d.id).await.unwrap_err().status(), 409);
    let d2 = w
        .downloads
        .create(&req("org/m", "spark", Some("hub")))
        .await
        .unwrap();
    assert_eq!(w.downloads.cancel(&d2.id).await.unwrap().state, "cancelled");
    assert_eq!(w.downloads.resume(&d2.id).await.unwrap_err().status(), 409);
    w.stop().await;
}

#[tokio::test]
async fn test_shutdown_parks_running_downloads() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(t.path(), None, WorldOpts::default()).await;
    w.hub.st().gate = Some(Gate::new());
    let d = w
        .downloads
        .create(&req("org/m", "spark", Some("hf")))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    w.downloads.shutdown().await;
    assert_eq!(w.downloads.get(&d.id).unwrap().state, "paused");
    w.stop().await;
}

// ---- progress -------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_progress_is_published_on_the_bus_with_speed_and_eta() {
    let t = tempfile::tempdir().unwrap();
    let (w, _) = make(
        t.path(),
        None,
        WorldOpts {
            clock: Some(tick()),
            ..WorldOpts::default()
        },
    )
    .await;
    let mut sub = w.bus.subscribe("downloads");
    let d = run(&w, "hf").await;
    let events = drain(&mut sub).await;
    assert!(d.state == "completed" && !events.is_empty());
    assert!(events.iter().all(|e| e["host_id"] == "spark"
        && e["root_id"] == "hf"
        && e["kind"] == "hf"
        && e["id"] == d.id.as_str()));
    let running: Vec<&Value> = events
        .iter()
        .filter(|e| e["state"] == "running" && e["done_bytes"].as_i64().unwrap() > 0)
        .collect();
    let dones: Vec<i64> = running
        .iter()
        .map(|e| e["done_bytes"].as_i64().unwrap())
        .collect();
    let mut sorted = dones.clone();
    sorted.sort_unstable();
    assert!(!running.is_empty() && dones == sorted);
    assert!(
        running
            .iter()
            .any(|e| e["speed_bps"].as_f64().unwrap() > 0.0 && !e["eta_s"].is_null())
    );
    assert!(
        events
            .iter()
            .map(|e| e["done_bytes"].as_i64().unwrap())
            .max()
            .unwrap()
            <= TOTAL
    );
    let last = events.last().unwrap();
    assert!(last["state"] == "completed" && last["done_bytes"] == TOTAL);
    assert!(running.iter().any(|e| e["current_file"] == "w.safetensors"));
    w.stop().await;
}

#[tokio::test]
async fn test_progress_counts_a_resumed_partial_as_done() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(
        t.path(),
        None,
        WorldOpts {
            clock: Some(tick()),
            ..WorldOpts::default()
        },
    )
    .await;
    seed_partial(&nd, "hf", &rel_w(), &PAYLOAD[..20]);
    let mut sub = w.bus.subscribe("downloads");
    run(&w, "hf").await;
    let events = drain(&mut sub).await;
    let first = events.iter().find(|e| e["state"] == "running").unwrap();
    assert!(
        first["done_bytes"] == 20 && first["total_bytes"] == TOTAL,
        "{first}"
    );
    w.stop().await;
}

// ---- other machines ---------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_downloads_go_to_whichever_machine_is_named() {
    let t = tempfile::tempdir().unwrap();
    let a = spark(t.path(), NodeOpts::default());
    let b = node(
        t.path(),
        "gpu2",
        NodeOpts {
            label: Some("GPU box".into()),
            roots: vec![RootCfg::new("models", "hf-hub")],
            ..NodeOpts::default()
        },
    );
    let w = World::new(
        t.path(),
        vec![Arc::clone(&a), Arc::clone(&b)],
        WorldOpts::default(),
    );
    w.hub.add_files("org/m", &files(), None);
    let w = w.start().await;
    let d1 = run_req(&w, req("org/m", "gpu2", None)).await;
    assert!(
        d1.state == "completed" && d1.root_id.as_deref() == Some("models") && d1.host_id == "gpu2"
    );
    assert!(
        b.root_path("models")
            .join("models--org--m/refs/main")
            .is_file()
    );
    let mut blobs = Vec::new();
    collect(&a.base.join("hf"), &mut blobs, "");
    assert!(blobs.is_empty());
    assert!(!a.calls().iter().any(|(op, _)| op.starts_with("fs.")));
    w.settle().await;
    assert!(
        w.library
            .find_copy("gpu2", &ModelRef::hf("org/m"), false)
            .is_some()
    );
    assert!(
        w.library
            .find_copy("spark", &ModelRef::hf("org/m"), false)
            .is_none()
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_node_is_only_ever_asked_for_file_and_library_operations() {
    let t = tempfile::tempdir().unwrap();
    let (w, nd) = make(t.path(), None, WorldOpts::default()).await;
    run(&w, "hf").await;
    let allowed = [
        "models.list",
        "models.scan",
        "fs.stat",
        "fs.write",
        "fs.symlink",
        "fs.put_small",
    ];
    assert!(
        nd.calls()
            .iter()
            .all(|(op, _)| allowed.contains(&op.as_str()))
    );
    let text = format!("{:?}", nd.calls());
    assert!(
        !text.contains("huggingface")
            && !text.contains("http")
            && !text.contains("Authorization")
            && !text.contains("Bearer")
    );
    w.stop().await;
}
