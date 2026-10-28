//! Port of backend/tests/data/test_copy.py: copies between machines and drives (a library copy's files go source node
//! -> hub -> target node, with the HF layout preserved where both ends are HF caches).

#![allow(clippy::unwrap_used)]

#[path = "common/wsc_data.rs"]
mod wsc_data;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ec_hub::domain::errors::{AppError, NodeError, upstream_error};
use ec_hub::domain::fleet::{CopyRequest, LibraryCopy};
use ec_hub::domain::models::{Download, DownloadRequest};
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::downloads::{CopyKind, classify_copy_source};
use ec_hub::services::library::CopySource;
use serde_json::{Value, json};
use wsc_data::*;

fn machines(tmp: &Path) -> (Arc<DataNode>, Arc<DataNode>) {
    let src = node(
        tmp,
        "alpha",
        NodeOpts {
            label: Some("Alpha".into()),
            addresses: vec!["10.0.0.1".into()],
            roots: vec![
                RootCfg::new("hf", "hf-home"),
                RootCfg::new("hf2", "hf-hub"),
                RootCfg::new("gguf", "gguf"),
                RootCfg::new("pl", "plain"),
                RootCfg::new("ro", "hf-home").ro(),
            ],
            ..NodeOpts::default()
        },
    );
    let dst = node(
        tmp,
        "beta",
        NodeOpts {
            label: Some("Beta".into()),
            addresses: vec!["10.0.0.2".into()],
            roots: vec![
                RootCfg::new("hub", "hf-hub"),
                RootCfg::new("home", "hf-home"),
                RootCfg::new("gguf", "gguf"),
                RootCfg::new("pl", "plain"),
                RootCfg::new("ro", "hf-hub").ro(),
            ],
            ..NodeOpts::default()
        },
    );
    (src, dst)
}

async fn world(tmp: &Path, seed: bool) -> (World, Arc<DataNode>, Arc<DataNode>) {
    let (src, dst) = machines(tmp);
    if seed {
        seed_hf_cache(
            &src.root_path("hf"),
            "org/m",
            &files(),
            "hf-home",
            COMMIT,
            "main",
        );
        src.scan();
    }
    let w = World::new(
        tmp,
        vec![Arc::clone(&src), Arc::clone(&dst)],
        WorldOpts::default(),
    );
    w.hub.add_files("org/m", &files(), None);
    (w.start().await, src, dst)
}

fn source_copy(w: &World) -> String {
    w.library
        .find_copy("alpha", &ModelRef::hf("org/m"), false)
        .unwrap()
        .id
}

async fn copy(w: &World, cid: &str, host: &str, root: Option<&str>) -> Download {
    let d = w
        .downloads
        .copy(
            cid,
            &CopyRequest {
                host_id: host.into(),
                root_id: root.map(str::to_string),
            },
        )
        .await
        .unwrap();
    w.downloads.wait(&d.id).await;
    w.downloads.get(&d.id).unwrap()
}

fn pe(e: &AppError) -> (u16, String, String) {
    (
        e.status(),
        e.code().to_string(),
        e.problem().map(|p| p.detail.clone()).unwrap_or_default(),
    )
}

fn reads(nd: &DataNode) -> Vec<Value> {
    nd.reads()
}

fn names_in(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    v.sort();
    v
}

// ---- Hugging Face cache directories ------------------------------------------------------------------------------
#[tokio::test]
async fn test_copy_between_machines_keeps_the_hf_layout_and_moves_the_big_files_machine_to_machine()
{
    let t = tempfile::tempdir().unwrap();
    let (w, src, dst) = world(t.path(), true).await;
    let cid = source_copy(&w);
    let d = copy(&w, &cid, "beta", Some("hub")).await;
    assert!(
        d.state == "completed"
            && d.kind == "copy"
            && d.source_copy_id.as_deref() == Some(cid.as_str())
    );
    assert_eq!(
        (
            d.host_id.as_str(),
            d.root_id.as_deref(),
            d.repo_id.as_str(),
            d.commit_sha.as_deref()
        ),
        ("beta", Some("hub"), "org/m", Some(COMMIT))
    );
    assert_eq!(
        (d.total_bytes, d.done_bytes, d.files_total, d.files_done),
        (TOTAL, TOTAL, 3, 3)
    );
    let repo = dst.root_path("hub").join("models--org--m");
    assert_eq!(
        std::fs::read(repo.join("blobs").join(sha())).unwrap(),
        PAYLOAD
    );
    let link = repo.join(format!("snapshots/{COMMIT}/w.safetensors"));
    assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
    assert_eq!(
        std::fs::read_link(&link).unwrap().to_string_lossy(),
        format!("../../blobs/{}", sha())
    );
    assert_eq!(
        std::fs::read(repo.join(format!("snapshots/{COMMIT}/config.json"))).unwrap(),
        b"{}"
    );
    assert_eq!(
        std::fs::read(repo.join(format!("snapshots/{COMMIT}/notes.txt"))).unwrap(),
        b"n"
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("refs/main")).unwrap(),
        COMMIT
    );
    assert_eq!(
        reads(&src),
        [json!({"rel": format!("hub/models--org--m/blobs/{}", sha()), "offset": 0})]
    );
    let opened: Vec<String> = w.hub.opens().into_iter().map(|o| o.2).collect();
    assert!(!opened.contains(&"w.safetensors".to_string()));
    let mut set = opened.clone();
    set.sort();
    set.dedup();
    assert_eq!(set, ["config.json", "notes.txt"]);
    let wr = dst
        .writes()
        .into_iter()
        .find(|x| x["rel"].as_str().unwrap().ends_with(&sha()))
        .unwrap();
    assert_eq!(wr["sha256"], sha());
    w.settle().await;
    let hosts: Vec<String> = w.library.models(None, None, None).unwrap()[0]
        .copies
        .iter()
        .map(|c| c.host_id.clone())
        .collect();
    assert_eq!(hosts, ["alpha", "beta"]);
    w.stop().await;
}

#[tokio::test]
async fn test_copy_from_an_hf_home_into_an_hf_home_and_the_default_root() {
    let t = tempfile::tempdir().unwrap();
    let (w, _, dst) = world(t.path(), true).await;
    let d = w
        .downloads
        .copy(
            &source_copy(&w),
            &CopyRequest {
                host_id: "beta".into(),
                root_id: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(d.root_id.as_deref(), Some("hub"));
    w.downloads.wait(&d.id).await;
    let d2 = copy(&w, &source_copy(&w), "beta", Some("home")).await;
    assert!(
        d2.state == "completed"
            && dst
                .root_path("home")
                .join("hub/models--org--m/refs/main")
                .is_file()
    );
    w.stop().await;
}

#[tokio::test]
async fn test_copy_to_another_drive_of_the_same_machine() {
    let t = tempfile::tempdir().unwrap();
    let (w, src, _) = world(t.path(), true).await;
    let d = copy(&w, &source_copy(&w), "alpha", Some("hf2")).await;
    assert!(d.state == "completed" && d.host_id == "alpha" && d.root_id.as_deref() == Some("hf2"));
    assert_eq!(
        std::fs::read(
            src.root_path("hf2")
                .join(format!("models--org--m/blobs/{}", sha()))
        )
        .unwrap(),
        PAYLOAD
    );
    w.settle().await;
    let mut roots: Vec<String> = w
        .library
        .copies_of("alpha")
        .into_iter()
        .map(|c| c.root_id)
        .collect();
    roots.sort();
    assert_eq!(roots, ["hf", "hf2"]);
    w.stop().await;
}

#[tokio::test]
async fn test_copy_resumes_from_the_partial_on_the_target() {
    let t = tempfile::tempdir().unwrap();
    let (w, src, dst) = world(t.path(), true).await;
    let rel = format!("models--org--m/blobs/{}", sha());
    let part = partial_of(&dst, "hub", &rel);
    std::fs::create_dir_all(part.parent().unwrap()).unwrap();
    std::fs::write(&part, &PAYLOAD[..20]).unwrap();
    let d = copy(&w, &source_copy(&w), "beta", Some("hub")).await;
    assert_eq!(d.state, "completed");
    assert_eq!(
        reads(&src),
        [json!({"rel": format!("hub/models--org--m/blobs/{}", sha()), "offset": 20})]
    );
    assert_eq!(
        dst.writes()
            .into_iter()
            .find(|x| x["rel"] == rel.as_str())
            .unwrap()["offset"],
        20
    );
    assert_eq!(
        std::fs::read(dst.root_path("hub").join(&rel)).unwrap(),
        PAYLOAD
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_copy_whose_bytes_change_in_transit_fails_the_nodes_check_and_can_be_retried() {
    let t = tempfile::tempdir().unwrap();
    let (w, src, dst) = world(t.path(), true).await;
    let blob = src
        .root_path("hf")
        .join(format!("hub/models--org--m/blobs/{}", sha()));
    std::fs::write(&blob, [b'X'; 50]).unwrap();
    let d = copy(&w, &source_copy(&w), "beta", Some("hub")).await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("hash_mismatch"));
    assert!(d.error.as_deref().unwrap().starts_with("Beta: "));
    assert!(!partial_of(&dst, "hub", &format!("models--org--m/blobs/{}", sha())).exists());
    std::fs::write(&blob, PAYLOAD).unwrap();
    w.downloads.resume(&d.id).await.unwrap();
    w.downloads.wait(&d.id).await;
    assert_eq!(w.downloads.get(&d.id).unwrap().state, "completed");
    w.stop().await;
}

#[tokio::test]
async fn test_a_read_error_on_the_source_machine_fails_the_copy_naming_that_machine() {
    let t = tempfile::tempdir().unwrap();
    let (w, src, _) = world(t.path(), true).await;
    let cid = source_copy(&w);
    src.st().read_error = Some(NodeError::new("alpha", "io_error", "the disk is failing"));
    let d = copy(&w, &cid, "beta", Some("hub")).await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("io_error"));
    assert!(d.error.as_deref().unwrap().starts_with("Alpha: "));
    w.stop().await;
}

#[tokio::test]
async fn test_a_source_blob_that_is_missing_means_the_copy_is_not_of_that_revision() {
    let t = tempfile::tempdir().unwrap();
    let (w, src, dst) = world(t.path(), true).await;
    std::fs::remove_file(
        src.root_path("hf")
            .join(format!("hub/models--org--m/blobs/{}", sha())),
    )
    .unwrap();
    let d = copy(&w, &source_copy(&w), "beta", Some("hub")).await;
    assert!(
        d.state == "failed"
            && d.error_code.as_deref() == Some("copy_source_mismatch")
            && dst.writes().is_empty()
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_copy_needs_the_hubs_listing_and_says_so_when_hugging_face_is_out_of_reach() {
    let t = tempfile::tempdir().unwrap();
    let (w, _, dst) = world(t.path(), true).await;
    let cid = source_copy(&w);
    w.hub.st().model_override = Some(Arc::new(|_| {
        Err(upstream_error("cannot reach Hugging Face: ConnectError").into())
    }));
    let d = copy(&w, &cid, "beta", Some("hub")).await;
    assert!(
        d.state == "failed"
            && d.error_code.as_deref() == Some("upstream_error")
            && dst.writes().is_empty()
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_copy_of_a_model_with_only_some_files_stays_partial() {
    let t = tempfile::tempdir().unwrap();
    let (w, _, dst) = world(t.path(), true).await;
    let mut f = files();
    f.push(("extra.safetensors".into(), vec![b'E'; 30]));
    w.hub.add_files("org/m", &f, None);
    let d = copy(&w, &source_copy(&w), "beta", Some("hub")).await;
    assert!(d.state == "completed" && d.files_total == 3);
    let snaps = dst
        .root_path("hub")
        .join(format!("models--org--m/snapshots/{COMMIT}"));
    assert_eq!(
        names_in(&snaps),
        ["config.json", "notes.txt", "w.safetensors"]
    );
    w.stop().await;
}

#[tokio::test]
async fn test_copy_into_a_gguf_root_flattens_the_snapshot() {
    let t = tempfile::tempdir().unwrap();
    let (w, _, dst) = world(t.path(), true).await;
    let d = copy(&w, &source_copy(&w), "beta", Some("gguf")).await;
    assert_eq!(d.state, "completed");
    let target = dst.root_path("gguf").join("org--m");
    assert_eq!(
        names_in(&target),
        ["config.json", "notes.txt", "w.safetensors"]
    );
    assert_eq!(
        std::fs::read(target.join("w.safetensors")).unwrap(),
        PAYLOAD
    );
    assert!(
        !target
            .join("w.safetensors")
            .symlink_metadata()
            .unwrap()
            .file_type()
            .is_symlink()
    );
    w.stop().await;
}

// ---- one quant of a GGUF repo, and loose .gguf files ------------------------------------------------------------
#[tokio::test]
async fn test_a_gguf_quant_in_an_hf_cache_copies_only_its_own_files() {
    let t = tempfile::tempdir().unwrap();
    let (q4, q8) = (b"q4".repeat(20), b"q8".repeat(30));
    let files: Vec<(String, Vec<u8>)> = vec![
        ("m-Q4_K_M.gguf".into(), q4.clone()),
        ("m-Q8_0.gguf".into(), q8),
        ("README.md".into(), b"hi".to_vec()),
    ];
    let (src, dst) = machines(t.path());
    seed_hf_cache(
        &src.root_path("hf"),
        "org/g",
        &files,
        "hf-home",
        COMMIT,
        "main",
    );
    {
        let mut st = src.st();
        st.scan_disk_enabled = false;
        st.entries = vec![gguf_entry(
            "hf",
            &format!("hub/models--org--g/snapshots/{COMMIT}/m-Q4_K_M.gguf"),
            json!({"name": "org/g", "quant": "Q4_K_M", "size": q4.len()}),
        )];
    }
    let w = World::new(
        t.path(),
        vec![Arc::clone(&src), Arc::clone(&dst)],
        WorldOpts::default(),
    );
    w.hub.add_files("org/g", &files, None);
    let w = w.start().await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    let d = copy(&w, &cid, "beta", Some("hub")).await;
    assert!(d.state == "completed" && d.files_total == 1 && d.total_bytes == q4.len() as i64);
    let snaps = dst
        .root_path("hub")
        .join(format!("models--org--g/snapshots/{COMMIT}"));
    assert_eq!(names_in(&snaps), ["m-Q4_K_M.gguf"]);
    assert_eq!(std::fs::read(snaps.join("m-Q4_K_M.gguf")).unwrap(), q4);
    w.stop().await;
}

#[tokio::test]
async fn test_a_loose_gguf_file_and_every_shard_of_a_split_set_are_copied_as_they_are() {
    let t = tempfile::tempdir().unwrap();
    let (src, dst) = machines(t.path());
    let shards: Vec<(String, Vec<u8>)> = (1u8..=3)
        .map(|i| {
            (
                format!("models/big-Q4_K_M-0000{i}-of-00003.gguf"),
                vec![i; 40 + usize::from(i)],
            )
        })
        .collect();
    let single = vec![b's'; 25];
    for (rel, data) in shards.iter().chain(std::iter::once(&(
        "models/one-Q8_0.gguf".to_string(),
        single.clone(),
    ))) {
        let p = src.root_path("gguf").join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, data).unwrap();
    }
    let total: usize = shards.iter().map(|(_, d)| d.len()).sum();
    let first = "models/big-Q4_K_M-00001-of-00003.gguf";
    {
        let mut st = src.st();
        st.scan_disk_enabled = false;
        st.entries = vec![
            gguf_entry(
                "gguf",
                first,
                json!({"name": "big", "quant": "Q4_K_M", "size": total, "shards": 3}),
            ),
            gguf_entry(
                "gguf",
                "models/one-Q8_0.gguf",
                json!({"name": "one", "quant": "Q8_0", "size": single.len()}),
            ),
        ];
    }
    let w = World::new(
        t.path(),
        vec![Arc::clone(&src), Arc::clone(&dst)],
        WorldOpts::default(),
    )
    .start()
    .await;
    let gref = |rel: &str| ModelRef {
        format: "gguf".into(),
        repo_id: None,
        root_id: Some("gguf".into()),
        rel: Some(rel.into()),
        name: None,
    };
    let big = w.library.find_copy("alpha", &gref(first), false).unwrap();
    let one = w
        .library
        .find_copy("alpha", &gref("models/one-Q8_0.gguf"), false)
        .unwrap();
    let d = copy(&w, &big.id, "beta", Some("gguf")).await;
    assert!(d.state == "completed" && d.files_total == 3 && d.total_bytes == total as i64);
    for (rel, data) in &shards {
        assert_eq!(
            &std::fs::read(dst.root_path("gguf").join(rel)).unwrap(),
            data
        );
    }
    assert_eq!(
        reads(&src)
            .into_iter()
            .map(|x| x["rel"].as_str().unwrap().to_string())
            .collect::<Vec<_>>(),
        shards.iter().map(|(r, _)| r.clone()).collect::<Vec<_>>()
    );
    assert!(dst.writes().iter().all(|x| x["sha256"].is_null()));
    let d2 = copy(&w, &one.id, "beta", None).await;
    assert!(d2.state == "completed" && d2.root_id.as_deref() == Some("gguf"));
    assert_eq!(
        std::fs::read(dst.root_path("gguf").join("models/one-Q8_0.gguf")).unwrap(),
        single
    );
    assert_eq!(w.hub.st().listings, 0);
    w.stop().await;
}

#[tokio::test]
async fn test_a_split_set_with_a_missing_shard_is_not_copied_half() {
    let t = tempfile::tempdir().unwrap();
    let (src, dst) = machines(t.path());
    std::fs::write(
        src.root_path("gguf").join("x-Q4_K_M-00001-of-00002.gguf"),
        [b'1'; 10],
    )
    .unwrap();
    {
        let mut st = src.st();
        st.scan_disk_enabled = false;
        st.entries = vec![gguf_entry(
            "gguf",
            "x-Q4_K_M-00001-of-00002.gguf",
            json!({"name": "x", "size": 10, "shards": 2}),
        )];
    }
    let w = World::new(t.path(), vec![src, Arc::clone(&dst)], WorldOpts::default())
        .start()
        .await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    let d = copy(&w, &cid, "beta", Some("gguf")).await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("copy_source_mismatch"));
    assert!(d.error.as_deref().unwrap().contains("shard") && dst.writes().is_empty());
    w.stop().await;
}

// ---- refusals --------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_what_cannot_be_copied_says_why() {
    let t = tempfile::tempdir().unwrap();
    let (src, dst) = machines(t.path());
    {
        let mut st = src.st();
        st.scan_disk_enabled = false;
        st.entries = vec![
            entry("pl", "x/manifests", "ollama", Some("q:1"), json!({})),
            entry("pl", "other", "other", Some("o"), json!({})),
            entry(
                "pl",
                "uncounted",
                "plain",
                Some("uncounted"),
                json!({"files": null}),
            ),
            entry(
                "pl",
                "bounded",
                "plain",
                Some("bounded"),
                json!({"files": 5, "warnings": ["size is a lower bound: the scan budget ran out"]}),
            ),
            gguf_entry(
                "hf",
                "hub/models--org--bad--name/snapshots/x/y.gguf",
                json!({}),
            ),
            hf_entry("hf", "org/ok", json!({})),
        ];
    }
    let w = World::new(t.path(), vec![src, dst], WorldOpts::default())
        .start()
        .await;
    let refused: Vec<LibraryCopy> = w
        .library
        .copies_of("alpha")
        .into_iter()
        .filter(|c| c.rel != "hub/models--org--ok")
        .collect();
    let mut rels: Vec<String> = refused.iter().map(|c| c.rel.clone()).collect();
    rels.sort();
    assert_eq!(
        rels,
        [
            "bounded",
            "hub/models--org--bad--name/snapshots/x/y.gguf",
            "other",
            "uncounted",
            "x/manifests"
        ]
    );
    for c in &refused {
        let e = w
            .downloads
            .copy(
                &c.id,
                &CopyRequest {
                    host_id: "beta".into(),
                    root_id: None,
                },
            )
            .await
            .unwrap_err();
        let (s, code, detail) = pe(&e);
        assert!(
            s == 501 && code == "copy_unsupported" && detail.contains("directory listing"),
            "{}",
            c.rel
        );
    }
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM downloads"), 0);
    w.stop().await;
}

// ---- plain model directories ------------------------------------------------------------------------------------------
fn plain_files() -> Vec<(String, Vec<u8>)> {
    let index = serde_json::to_vec(
        &json!({"metadata": {}, "weight_map": {"a": "model-00001-of-00002.safetensors",
        "b": "model-00002-of-00002.safetensors", "c": "model-00001-of-00002.safetensors"}}),
    )
    .unwrap();
    let index = String::from_utf8(index)
        .unwrap()
        .replace(':', ": ")
        .replace(',', ", ")
        .into_bytes();
    vec![
        ("config.json".into(), br#"{"model_type": "x"}"#.to_vec()),
        ("tokenizer.json".into(), b"{}".to_vec()),
        ("README.md".into(), b"# hi".to_vec()),
        ("model-00001-of-00002.safetensors".into(), vec![b'a'; 70]),
        ("model-00002-of-00002.safetensors".into(), vec![b'b'; 55]),
        ("model.safetensors.index.json".into(), index),
    ]
}

fn plain_world(
    tmp: &Path,
    files: &[(String, Vec<u8>)],
    rel: &str,
) -> (World, Arc<DataNode>, Arc<DataNode>) {
    let (src, dst) = machines(tmp);
    for (name, data) in files {
        let p = src.root_path("pl").join(rel).join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, data).unwrap();
    }
    let size: usize = files.iter().map(|(_, d)| d.len()).sum();
    {
        let mut st = src.st();
        st.scan_disk_enabled = false;
        st.entries = vec![entry(
            "pl",
            rel,
            "plain",
            Some(rel.rsplit('/').next().unwrap()),
            json!({"size": size, "files": files.len()}),
        )];
    }
    (
        World::new(
            tmp,
            vec![Arc::clone(&src), Arc::clone(&dst)],
            WorldOpts::default(),
        ),
        src,
        dst,
    )
}

#[tokio::test]
async fn test_a_plain_directory_is_copied_when_every_file_is_accounted_for() {
    let t = tempfile::tempdir().unwrap();
    let pf = plain_files();
    let (w, src, dst) = plain_world(t.path(), &pf, "mymodel");
    let w = w.start().await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    let d = copy(&w, &cid, "beta", Some("pl")).await;
    assert_eq!(d.state, "completed", "{:?} {:?}", d.error_code, d.error);
    let total: usize = pf.iter().map(|(_, d)| d.len()).sum();
    assert!(
        d.files_total == 6 && d.total_bytes == total as i64 && d.root_id.as_deref() == Some("pl")
    );
    for (name, data) in &pf {
        assert_eq!(
            &std::fs::read(dst.root_path("pl").join("mymodel").join(name)).unwrap(),
            data
        );
    }
    let mut read: Vec<String> = reads(&src)
        .into_iter()
        .map(|x| x["rel"].as_str().unwrap().to_string())
        .collect();
    read.sort();
    read.dedup();
    let mut want: Vec<String> = pf.iter().map(|(n, _)| format!("mymodel/{n}")).collect();
    want.sort();
    assert_eq!(read, want);
    assert!(w.hub.st().listings == 0 && !dst.calls().iter().any(|(op, _)| op == "models.config"));
    w.stop().await;
}

#[tokio::test]
async fn test_a_plain_directory_with_a_file_the_probe_cannot_name_is_refused_with_the_numbers() {
    let t = tempfile::tempdir().unwrap();
    let mut pf = plain_files();
    pf.push(("weird-name.dat".into(), vec![b'?'; 9]));
    let (w, _, dst) = plain_world(t.path(), &pf, "mymodel");
    let w = w.start().await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    let d = copy(&w, &cid, "beta", Some("pl")).await;
    let err = d.error.clone().unwrap_or_default();
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("copy_listing_incomplete"));
    let base: usize = plain_files().iter().map(|(_, d)| d.len()).sum();
    assert!(err.contains("found 6 of the directory's 7 files"), "{err}");
    assert!(
        err.contains(&format!("{base} of {} bytes", base + 9)),
        "{err}"
    );
    assert!(dst.writes().is_empty());
    w.stop().await;
}

#[tokio::test]
async fn test_a_plain_directory_with_a_subfolder_is_refused() {
    let t = tempfile::tempdir().unwrap();
    let mut pf = plain_files();
    pf.push(("1_Pooling/config.json".into(), b"{}".to_vec()));
    let (w, _, dst) = plain_world(t.path(), &pf, "mymodel");
    let w = w.start().await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    let d = copy(&w, &cid, "beta", Some("pl")).await;
    assert!(
        d.state == "failed"
            && d.error_code.as_deref() == Some("copy_listing_incomplete")
            && dst.writes().is_empty()
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_directory_that_changed_since_the_scan_is_refused() {
    let t = tempfile::tempdir().unwrap();
    let (w, src, dst) = plain_world(t.path(), &plain_files(), "mymodel");
    let w = w.start().await;
    std::fs::write(
        src.root_path("pl").join("mymodel/README.md"),
        b"# a much longer readme than the scan saw",
    )
    .unwrap();
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    let d = copy(&w, &cid, "beta", Some("pl")).await;
    assert!(
        d.state == "failed"
            && d.error_code.as_deref() == Some("copy_listing_incomplete")
            && dst.writes().is_empty()
    );
    w.stop().await;
}

#[tokio::test]
async fn test_hostile_names_in_a_weight_index_are_never_sent_to_a_node() {
    let t = tempfile::tempdir().unwrap();
    let index = serde_json::to_vec(
        &json!({"weight_map": {"a": "../evil.bin", "b": "/abs/path", "c": 5, "d": "a\\b",
        "e": "ok.safetensors", "f": "x/./y"}}),
    )
    .unwrap();
    let files: Vec<(String, Vec<u8>)> = vec![
        ("config.json".into(), b"{}".to_vec()),
        ("ok.safetensors".into(), vec![b'o'; 33]),
        ("model.safetensors.index.json".into(), index),
    ];
    let (w, src, _) = plain_world(t.path(), &files, "mymodel");
    let w = w.start().await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    let d = copy(&w, &cid, "beta", Some("pl")).await;
    assert_eq!(d.state, "completed");
    let statted: Vec<String> = src
        .ops("fs.stat")
        .into_iter()
        .flat_map(|a| {
            a["paths"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(statted.contains(&"mymodel/ok.safetensors".to_string()));
    assert!(
        !statted.iter().any(|p| p.contains("..")
            || p.starts_with('/')
            || p.contains('\\')
            || p.contains("/./"))
    );
    w.stop().await;
}

#[tokio::test]
async fn test_an_oversized_or_broken_weight_index_is_handled() {
    let t = tempfile::tempdir().unwrap();
    let mut big = br#"{"weight_map": {"a": "x.safetensors"}, "pad": ""#.to_vec();
    big.extend(vec![b'p'; 5 * 1024 * 1024]);
    big.extend(br#""}"#);
    let files = vec![
        ("config.json".to_string(), b"{}".to_vec()),
        ("model.safetensors.index.json".to_string(), big),
    ];
    let (w, _, dst) = plain_world(t.path(), &files, "mymodel");
    let w = w.start().await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    let d = copy(&w, &cid, "beta", Some("pl")).await;
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("copy_listing_incomplete"));
    assert!(d.error.as_deref().unwrap().contains("4 MiB") && dst.writes().is_empty());
    w.stop().await;
    let x = t.path().join("x");
    let files2 = vec![
        ("config.json".to_string(), b"{}".to_vec()),
        (
            "model.safetensors.index.json".to_string(),
            b"not json at all".to_vec(),
        ),
    ];
    let (w2, _, _) = plain_world(&x, &files2, "mymodel");
    let w2 = w2.start().await;
    let cid2 = w2.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    assert_eq!(
        copy(&w2, &cid2, "beta", Some("pl")).await.state,
        "completed"
    );
    w2.stop().await;
}

#[tokio::test]
async fn test_a_directory_that_is_itself_the_model_root_and_names_with_odd_characters() {
    let t = tempfile::tempdir().unwrap();
    let (src, dst) = machines(t.path());
    let whole_dir = src.base.join("whole");
    std::fs::create_dir_all(&whole_dir).unwrap();
    src.add_root(RootCfg::new("whole", "plain").at(&whole_dir));
    std::fs::write(whole_dir.join("config.json"), b"{}").unwrap();
    std::fs::write(whole_dir.join("pytorch_model.bin"), [b'w'; 7]).unwrap();
    let odd = src.root_path("pl").join("a b/c d");
    std::fs::create_dir_all(&odd).unwrap();
    std::fs::write(odd.join("config.json"), b"").unwrap();
    {
        let mut st = src.st();
        st.scan_disk_enabled = false;
        st.entries = vec![
            entry(
                "whole",
                "",
                "plain",
                Some("root model!"),
                json!({"size": 9, "files": 2}),
            ),
            entry(
                "pl",
                "a b/c d",
                "plain",
                Some("c d"),
                json!({"size": 0, "files": 1}),
            ),
        ];
    }
    let w = World::new(
        t.path(),
        vec![Arc::clone(&src), Arc::clone(&dst)],
        WorldOpts::default(),
    )
    .start()
    .await;
    let whole = w
        .library
        .copies_of("alpha")
        .into_iter()
        .find(|c| c.rel.is_empty())
        .unwrap();
    let spaced = w
        .library
        .copies_of("alpha")
        .into_iter()
        .find(|c| c.rel == "a b/c d")
        .unwrap();
    assert_eq!(whole.path, whole_dir.display().to_string());
    let d = copy(&w, &whole.id, "beta", Some("pl")).await;
    assert_eq!(d.state, "completed", "{:?} {:?}", d.error_code, d.error);
    assert_eq!(
        std::fs::read(dst.root_path("pl").join("root_model_/config.json")).unwrap(),
        b"{}"
    );
    let d2 = copy(&w, &spaced.id, "beta", Some("pl")).await;
    assert!(d2.state == "completed" && dst.root_path("pl").join("c_d/config.json").is_file());
    w.stop().await;
}

#[tokio::test]
async fn test_plain_directories_go_only_into_plain_or_auto_roots_and_only_those_take_them() {
    let t = tempfile::tempdir().unwrap();
    let (w, _, dst) = plain_world(t.path(), &plain_files(), "mymodel");
    let auto_dir = dst.base.join("auto");
    std::fs::create_dir_all(&auto_dir).unwrap();
    dst.add_root(RootCfg::new("auto", "auto").at(&auto_dir));
    let w = w.start().await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    for root in ["hub", "home", "gguf", "ro"] {
        let e = w
            .downloads
            .copy(
                &cid,
                &CopyRequest {
                    host_id: "beta".into(),
                    root_id: Some(root.into()),
                },
            )
            .await
            .unwrap_err();
        assert!(
            e.code() == "root_kind_unsupported" || e.code() == "root_not_writable",
            "{root}"
        );
    }
    let d = w
        .downloads
        .copy(
            &cid,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(d.root_id.as_deref(), Some("pl"));
    w.downloads.wait(&d.id).await;
    let d2 = w
        .downloads
        .copy(
            &cid,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("auto".into()),
            },
        )
        .await
        .unwrap();
    w.downloads.wait(&d2.id).await;
    assert!(
        w.downloads.get(&d2.id).unwrap().state == "completed"
            && auto_dir.join("mymodel/config.json").is_file()
    );
    let r = DownloadRequest {
        repo_id: "org/m".into(),
        revision: None,
        allow_patterns: None,
        host_id: "beta".into(),
        root_id: Some("pl".into()),
    };
    assert_eq!(
        w.downloads.create(&r).await.unwrap_err().code(),
        "root_kind_unsupported"
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_loose_gguf_can_go_into_an_empty_auto_root() {
    let t = tempfile::tempdir().unwrap();
    let (src, dst) = machines(t.path());
    std::fs::write(src.root_path("gguf").join("m-Q4_K_M.gguf"), [b'1'; 10]).unwrap();
    src.scan();
    let auto_dir = dst.base.join("auto");
    std::fs::create_dir_all(&auto_dir).unwrap();
    dst.add_root(RootCfg::new("auto", "auto").at(&auto_dir));
    let w = World::new(t.path(), vec![src, dst], WorldOpts::default())
        .start()
        .await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    let d = copy(&w, &cid, "beta", Some("auto")).await;
    assert!(
        d.state == "completed"
            && std::fs::read(auto_dir.join("m-Q4_K_M.gguf")).unwrap() == [b'1'; 10]
    );
    w.stop().await;
}

#[tokio::test]
async fn test_copy_refusals_at_create() {
    let t = tempfile::tempdir().unwrap();
    let (w, src, dst) = world(t.path(), true).await;
    let cid = source_copy(&w);
    let refused = |host: &'static str, root: Option<&'static str>, id: Option<String>| {
        let (w, cid) = (&w, cid.clone());
        async move {
            let e = w
                .downloads
                .copy(
                    &id.unwrap_or(cid),
                    &CopyRequest {
                        host_id: host.into(),
                        root_id: root.map(str::to_string),
                    },
                )
                .await
                .unwrap_err();
            pe(&e)
        }
    };
    let (s, c, _) = refused("beta", None, Some("f".repeat(16))).await;
    assert_eq!((s, c.as_str()), (404, "copy_not_found"));
    let (s, c, _) = refused("alpha", Some("hf"), None).await;
    assert_eq!((s, c.as_str()), (409, "copy_same_location"));
    let (s, c, _) = refused("beta", Some("ro"), None).await;
    assert_eq!((s, c.as_str()), (409, "root_not_writable"));
    let (s, c, _) = refused("beta", Some("pl"), None).await;
    assert_eq!((s, c.as_str()), (422, "root_kind_unsupported"));
    let (s, c, _) = refused("nobody", None, None).await;
    assert_eq!((s, c.as_str()), (404, "unknown_host"));
    dst.st().mode = "monitor".into();
    let (s, c, _) = refused("beta", None, None).await;
    assert_eq!((s, c.as_str()), (409, "monitor_only"));
    dst.st().mode = "manage".into();
    dst.st().state = "down".into();
    let (s, c, d) = refused("beta", None, None).await;
    assert!(s == 503 && c == "node_unavailable" && d.starts_with("Beta: "));
    dst.st().state = "up".into();
    src.st().state = "down".into();
    let (s, c, d) = refused("beta", None, None).await;
    assert!(s == 503 && c == "node_unavailable" && d.starts_with("Alpha: "));
    src.st().state = "up".into();
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM downloads"), 0);
    w.stop().await;
}

#[tokio::test]
async fn test_a_loose_gguf_file_only_goes_into_a_gguf_root() {
    let t = tempfile::tempdir().unwrap();
    let (src, dst) = machines(t.path());
    std::fs::write(src.root_path("gguf").join("m-Q4_K_M.gguf"), [b'1'; 10]).unwrap();
    src.scan();
    let w = World::new(t.path(), vec![src, dst], WorldOpts::default())
        .start()
        .await;
    let cid = w.library.models(None, None, None).unwrap()[0].copies[0]
        .id
        .clone();
    let e = w
        .downloads
        .copy(
            &cid,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("hub".into()),
            },
        )
        .await
        .unwrap_err();
    assert_eq!((e.status(), e.code()), (422, "root_kind_unsupported"));
    w.stop().await;
}

#[tokio::test]
async fn test_a_machine_that_already_sees_the_copy_over_nfs_can_still_copy_it_to_its_own_drive() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (nfs_alpha(t.path(), None), nfs_beta(t.path(), None));
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w = World::new(
        t.path(),
        vec![Arc::clone(&a), Arc::clone(&b)],
        WorldOpts::default(),
    );
    w.hub.add_files("org/m", &files(), None);
    let w = w.start().await;
    let c0 = w.library.models(None, None, None).unwrap()[0].copies[0].clone();
    assert_eq!(c0.visible_on, ["alpha", "beta"]);
    let e = w
        .downloads
        .copy(
            &c0.id,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("shared".into()),
            },
        )
        .await
        .unwrap_err();
    assert_eq!(e.code(), "copy_same_location");
    let src = w.library.source(&c0.id).unwrap();
    let mut locs = src.locations.clone();
    locs.sort();
    assert_eq!(
        locs,
        vec![
            ("alpha".to_string(), "hf".to_string()),
            ("beta".into(), "shared".into())
        ]
    );
    let d = w
        .downloads
        .copy(
            &c0.id,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("own".into()),
            },
        )
        .await
        .unwrap();
    assert!(
        d.root_id.as_deref() == Some("own")
            && d.host_id == "beta"
            && d.kind == "copy"
            && d.state == "queued"
    );
    w.downloads.cancel(&d.id).await.unwrap();
    assert!(
        ![&a, &b]
            .iter()
            .any(|n| n.calls().iter().any(|(op, _)| op.starts_with("fs.")))
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_copy_row_cannot_be_started_twice() {
    let t = tempfile::tempdir().unwrap();
    let (w, _, _) = world(t.path(), true).await;
    stop_midway(&w, 1);
    let cid = source_copy(&w);
    let d = w
        .downloads
        .copy(
            &cid,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("hub".into()),
            },
        )
        .await
        .unwrap();
    let e = w
        .downloads
        .copy(
            &cid,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("hub".into()),
            },
        )
        .await
        .unwrap_err();
    assert_eq!(e.code(), "download_exists");
    w.downloads
        .copy(
            &cid,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("home".into()),
            },
        )
        .await
        .unwrap();
    w.downloads.cancel(&d.id).await.unwrap();
    w.stop().await;
}

#[tokio::test]
async fn test_a_copy_whose_source_vanished_fails_clearly_when_resumed() {
    let t = tempfile::tempdir().unwrap();
    let (w, _, _) = world(t.path(), true).await;
    w.store
        .execute(
            "INSERT INTO downloads(id,repo_id,state,created_at,updated_at,host_id,root_id,kind,source_copy_id) \
             VALUES('dl_x','org/m','paused',1,1,'beta','hub','copy',?)",
            &[&"0".repeat(16)],
        )
        .unwrap();
    w.downloads.resume("dl_x").await.unwrap();
    w.downloads.wait("dl_x").await;
    let d = w.downloads.get("dl_x").unwrap();
    assert!(d.state == "failed" && d.error_code.as_deref() == Some("copy_source_gone"));
    w.stop().await;
}

// ---- pause, cancel and progress -----------------------------------------------------------------------------------------
#[tokio::test]
async fn test_a_copy_can_be_paused_and_resumed_midway_and_cancel_cleans_the_partial() {
    let t = tempfile::tempdir().unwrap();
    let (w, src, dst) = world(t.path(), true).await;
    src.st().read_chunk = 16;
    let cid = source_copy(&w);
    let rel = format!("models--org--m/blobs/{}", sha());
    let gate = Gate::new();
    src.st().read_gate = Some((2, gate.clone()));
    let d = w
        .downloads
        .copy(
            &cid,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("hub".into()),
            },
        )
        .await
        .unwrap();
    let p = partial_of(&dst, "hub", &rel);
    until(|| exists_with_size(&p, 32), "32 bytes").await;
    assert_eq!(w.downloads.pause(&d.id).await.unwrap().state, "paused");
    assert!(exists_with_size(&p, 32));
    gate.set();
    w.downloads.resume(&d.id).await.unwrap();
    w.downloads.wait(&d.id).await;
    assert_eq!(w.downloads.get(&d.id).unwrap().state, "completed");
    assert!(
        reads(&src)
            .contains(&json!({"rel": format!("hub/models--org--m/blobs/{}", sha()), "offset": 32}))
    );
    assert_eq!(
        std::fs::read(dst.root_path("hub").join(&rel)).unwrap(),
        PAYLOAD
    );
    w.stop().await;
}

#[tokio::test]
async fn test_copy_progress_events_say_it_is_a_copy() {
    let t = tempfile::tempdir().unwrap();
    let (w, _, _) = world(t.path(), true).await;
    let mut sub = w.bus.subscribe("downloads");
    let cid = source_copy(&w);
    let d = copy(&w, &cid, "beta", Some("hub")).await;
    let mut events = Vec::new();
    while let Ok(Some(ev)) = tokio::time::timeout(Duration::from_millis(20), sub.recv()).await {
        events.push(ev);
    }
    assert!(d.state == "completed" && !events.is_empty());
    assert!(
        events
            .iter()
            .all(|e| e["kind"] == "copy" && e["source_copy_id"] == cid.as_str())
    );
    let last = events.last().unwrap();
    assert!(last["state"] == "completed" && last["done_bytes"] == TOTAL);
    w.stop().await;
}

// ---- classification ------------------------------------------------------------------------------------------------------
#[test]
fn test_classify_copy_source() {
    let src = |rel: &str,
               fmt: &str,
               revision: Option<&str>,
               shards: Option<i64>,
               files: Option<i64>,
               warnings: Vec<&str>| CopySource {
        copy: LibraryCopy {
            id: "0".repeat(16),
            host_id: "h".into(),
            host_label: "H".into(),
            root_id: "r".into(),
            root_label: None,
            rel: rel.into(),
            path: format!("/x/{rel}"),
            size_bytes: 1,
            files,
            mtime: None,
            atime: None,
            drive: None,
            visible_on: Vec::new(),
            writable: false,
            deletable: false,
            in_use_by: Vec::new(),
        },
        format: fmt.into(),
        name: "n".into(),
        revision: revision.map(str::to_string),
        quant: None,
        shards,
        locations: vec![("h".into(), "r".into())],
        warnings: warnings.into_iter().map(str::to_string).collect(),
    };
    let hf = |repo: &str, rel: &str, rev: Option<&str>, only: Option<Vec<&str>>| CopyKind::Hf {
        repo_id: repo.into(),
        repo_rel: rel.into(),
        revision: rev.map(str::to_string),
        only: only.map(|o| o.into_iter().map(str::to_string).collect()),
    };
    assert_eq!(
        classify_copy_source(&src(
            "hub/models--org--m",
            "hf",
            Some(COMMIT),
            None,
            None,
            vec![]
        )),
        Some(hf("org/m", "hub/models--org--m", Some(COMMIT), None))
    );
    assert_eq!(
        classify_copy_source(&src(
            "models--org--m",
            "hf",
            Some("main"),
            None,
            None,
            vec![]
        )),
        Some(hf("org/m", "models--org--m", None, None))
    );
    let snap = classify_copy_source(&src(
        &format!("hub/models--org--m/snapshots/{COMMIT}/q/m-00001-of-00002.gguf"),
        "gguf",
        None,
        Some(2),
        None,
        vec![],
    ));
    assert_eq!(
        snap,
        Some(hf(
            "org/m",
            "hub/models--org--m",
            Some(COMMIT),
            Some(vec!["q/m-00001-of-00002.gguf", "q/m-00002-of-00002.gguf"])
        ))
    );
    assert!(
        classify_copy_source(&src(
            &format!("models--org--m/snapshots/{COMMIT}/model.safetensors"),
            "hf",
            None,
            None,
            None,
            vec![]
        ))
        .is_none()
    );
    assert!(
        classify_copy_source(&src("models--a--b--c", "hf", None, None, None, vec![])).is_none()
    );
    assert_eq!(
        classify_copy_source(&src("sub/m-Q4_K_M.gguf", "gguf", None, None, None, vec![])),
        Some(CopyKind::Gguf {
            rels: vec!["sub/m-Q4_K_M.gguf".into()]
        })
    );
    assert_eq!(
        classify_copy_source(&src(
            "m-00001-of-00002.gguf",
            "gguf",
            None,
            Some(2),
            None,
            vec![]
        )),
        Some(CopyKind::Gguf {
            rels: vec![
                "m-00001-of-00002.gguf".into(),
                "m-00002-of-00002.gguf".into()
            ]
        })
    );
    assert!(classify_copy_source(&src("sub/x.gguf", "plain", None, None, None, vec![])).is_none());
    let plain = |rel: &str| {
        Some(CopyKind::Plain {
            rel: rel.into(),
            name: "n".into(),
            files: 3,
            size: 1,
        })
    };
    assert_eq!(
        classify_copy_source(&src("dir", "plain", None, None, Some(3), vec![])),
        plain("dir")
    );
    assert_eq!(
        classify_copy_source(&src("", "plain", None, None, Some(3), vec![])),
        plain("")
    );
    assert!(
        classify_copy_source(&src(
            "dir",
            "plain",
            None,
            None,
            Some(3),
            vec!["size is a lower bound: x"]
        ))
        .is_none()
    );
    assert!(classify_copy_source(&src("../dir", "plain", None, None, Some(3), vec![])).is_none());
    assert!(classify_copy_source(&src("dir", "other", None, None, Some(3), vec![])).is_none());
    assert!(classify_copy_source(&src("../x.gguf", "gguf", None, None, None, vec![])).is_none());
}

#[test]
fn test_the_hash_helper_agrees_with_the_nodes() {
    use sha2::Digest as _;
    assert_eq!(hex::encode(sha2::Sha256::digest(PAYLOAD)), sha());
}

#[allow(dead_code)]
fn unused(_: Value) {}
