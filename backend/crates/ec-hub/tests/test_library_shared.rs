//! Port of backend/tests/data/test_library_shared.py: shared storage (docs/ARCHITECTURE.md 10.5) — a copy on host
//! B's NFS mount of host A's disk is A's copy, seen by B.

#![allow(clippy::unwrap_used)]

#[path = "common/wsc_data.rs"]
mod wsc_data;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ec_hub::domain::refs::ModelRef;
use ec_hub::services::store::Store;
use serde_json::{Value, json};
use wsc_data::*;

fn hf() -> ModelRef {
    ModelRef::hf("org/m")
}

fn gref(root: &str, rel: &str) -> ModelRef {
    ModelRef {
        format: "gguf".into(),
        repo_id: None,
        root_id: Some(root.into()),
        rel: Some(rel.into()),
        name: None,
    }
}

async fn world(tmp: &Path, nodes: Vec<Arc<DataNode>>) -> World {
    World::new(tmp, nodes, WorldOpts::default()).start().await
}

fn alpha(t: &Path) -> Arc<DataNode> {
    nfs_alpha(t, None)
}

fn beta(t: &Path) -> Arc<DataNode> {
    nfs_beta(t, None)
}

#[tokio::test]
async fn test_a_copy_on_an_nfs_mount_of_a_configured_host_is_that_hosts_copy() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (alpha(t.path()), beta(t.path()));
    a.st().entries = vec![hf_entry("hf", "org/m", json!({"size": 500}))];
    b.st().entries = vec![
        hf_entry("shared", "org/m", json!({"size": 500})),
        hf_entry_at("own", "org/own", false, json!({})),
    ];
    let w = world(t.path(), vec![a, b]).await;
    let lib = &w.library;
    let model = lib
        .models(None, None, None)
        .unwrap()
        .into_iter()
        .find(|m| m.name == "org/m")
        .unwrap();
    assert!(model.copies.len() == 1 && model.copies[0].host_id == "alpha");
    assert_eq!(model.copies[0].visible_on, ["alpha", "beta"]);
    assert_eq!(model.copies[0].path, "/data/hf/hub/models--org--m");
    assert_eq!(
        lib.copies_of("beta")
            .into_iter()
            .map(|c| c.rel)
            .collect::<Vec<_>>(),
        ["models--org--own"]
    );
    let seen = lib.visible_copies("beta");
    let mut rels: Vec<String> = seen.iter().map(|c| c.rel.clone()).collect();
    rels.sort();
    assert_eq!(rels, ["hub/models--org--m", "models--org--own"]);
    assert!(seen.iter().all(|c| c.host_id == "beta"));
    let shared = seen
        .iter()
        .find(|c| c.rel == "hub/models--org--m")
        .unwrap()
        .clone();
    assert_eq!(
        (
            shared.root_id.as_str(),
            shared.path.as_str(),
            shared.id.as_str()
        ),
        (
            "shared",
            "/mnt/alpha/hf/hub/models--org--m",
            model.copies[0].id.as_str()
        )
    );
    assert!(
        shared.visible_on == ["alpha", "beta"] && !shared.writable && shared.host_label == "Beta"
    );
    let d = shared.drive.clone().unwrap();
    assert!(d.label == "NFS from Alpha" && d.remote_host_id.as_deref() == Some("alpha"));
    let found = lib.find_copy("beta", &hf(), false).unwrap();
    assert_eq!(found, shared);
    let owner_view = lib.find_copy("alpha", &hf(), false).unwrap();
    assert_eq!(
        (
            owner_view.host_id.as_str(),
            owner_view.root_id.as_str(),
            owner_view.path.as_str()
        ),
        ("alpha", "hf", "/data/hf/hub/models--org--m")
    );
    assert!(owner_view.id == shared.id && !owner_view.drive.as_ref().unwrap().remote);
    assert!(lib.find_copy("beta", &hf(), true).is_none());
    assert_eq!(lib.find_copy("alpha", &hf(), true).unwrap(), owner_view);
    assert_eq!(
        lib.models(Some("beta"), None, None)
            .unwrap()
            .into_iter()
            .map(|m| m.name)
            .collect::<Vec<_>>(),
        ["org/m", "org/own"]
    );
    w.stop().await;
}

#[tokio::test]
async fn test_use_on_either_machine_marks_the_one_copy_in_use() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (alpha(t.path()), beta(t.path()));
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w = world(t.path(), vec![a, b]).await;
    w.set_busy(
        "beta",
        "/mnt/alpha/hf/hub/models--org--m",
        &["inst_on_beta"],
    );
    let c = w.library.models(None, None, None).unwrap()[0].copies[0].clone();
    assert!(c.in_use_by == ["inst_on_beta"] && !c.deletable);
    let e = w.library.delete_copy(&c.id).await.unwrap_err();
    assert!(e.code() == "model_in_use" && e.problem().unwrap().detail.contains("inst_on_beta"));
    w.set_busy("alpha", "/data/hf/hub/models--org--m", &["inst_on_alpha"]);
    let mut users = w.library.models(None, None, None).unwrap()[0].copies[0]
        .in_use_by
        .clone();
    users.sort();
    assert_eq!(users, ["inst_on_alpha", "inst_on_beta"]);
    w.stop().await;
}

#[tokio::test]
async fn test_gguf_copies_are_found_by_either_machines_coordinates() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (alpha(t.path()), beta(t.path()));
    a.st().entries = vec![gguf_entry("gguf", "a/x-Q4_K_M.gguf", json!({"name": "x"}))];
    b.st().entries = vec![gguf_entry(
        "shared_g",
        "a/x-Q4_K_M.gguf",
        json!({"name": "x"}),
    )];
    let w = world(t.path(), vec![a, b]).await;
    let models = w.library.models(None, None, None).unwrap();
    assert!(models.len() == 1 && models[0].copies[0].visible_on == ["alpha", "beta"]);
    let by_beta = w
        .library
        .find_copy("beta", &gref("shared_g", "a/x-Q4_K_M.gguf"), false)
        .unwrap();
    let by_owner = w
        .library
        .find_copy("beta", &gref("gguf", "a/x-Q4_K_M.gguf"), false)
        .unwrap();
    assert_eq!(by_owner, by_beta);
    assert_eq!(
        (
            by_beta.host_id.as_str(),
            by_beta.root_id.as_str(),
            by_beta.rel.as_str(),
            by_beta.path.as_str()
        ),
        (
            "beta",
            "shared_g",
            "a/x-Q4_K_M.gguf",
            "/mnt/alpha/gguf/a/x-Q4_K_M.gguf"
        )
    );
    assert_eq!(by_beta.id, models[0].copies[0].id);
    let on_alpha = w
        .library
        .find_copy("alpha", &gref("gguf", "a/x-Q4_K_M.gguf"), false)
        .unwrap();
    assert_eq!(
        (
            on_alpha.host_id.as_str(),
            on_alpha.root_id.as_str(),
            on_alpha.path.as_str()
        ),
        ("alpha", "gguf", "/data/gguf/a/x-Q4_K_M.gguf")
    );
    assert!(
        w.library
            .find_copy("alpha", &gref("shared_g", "a/x-Q4_K_M.gguf"), false)
            .is_none()
    );
    assert!(
        w.library
            .find_copy("beta", &gref("shared_g", "a/nope.gguf"), false)
            .is_none()
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_machine_with_two_roots_on_the_same_storage_gets_the_one_that_naturally_holds_the_model()
 {
    let t = tempfile::tempdir().unwrap();
    let a = alpha(t.path());
    let b = node(
        t.path(),
        "beta",
        NodeOpts {
            addresses: vec!["10.0.0.2".into()],
            hostname: Some("beta".into()),
            scan_disk: false,
            roots: vec![
                RootCfg::new("aaa_auto", "auto")
                    .ro()
                    .at("/mnt/alpha/hf/hub"),
                RootCfg::new("zzz_hub", "hf-hub")
                    .ro()
                    .at("/mnt/alpha/hf/hub"),
            ],
            mounts: Some(vec![
                mount_dict("/", MountOpts::default()),
                nfs_mount("10.0.0.1", "/data", "/mnt/alpha", &["aaa_auto", "zzz_hub"]),
            ]),
            ..NodeOpts::default()
        },
    );
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![
        hf_entry_at("aaa_auto", "org/m", false, json!({})),
        hf_entry_at("zzz_hub", "org/m", false, json!({})),
    ];
    let w = world(t.path(), vec![a, b]).await;
    assert_eq!(
        w.library.models(None, None, None).unwrap()[0].copies.len(),
        1
    );
    let found = w.library.find_copy("beta", &hf(), false).unwrap();
    assert!(found.root_id == "zzz_hub" && found.rel == "models--org--m");
    assert_eq!(
        w.library
            .visible_copies("beta")
            .into_iter()
            .map(|c| c.root_id)
            .collect::<Vec<_>>(),
        ["zzz_hub"]
    );
    let src = w.library.source(&found.id).unwrap();
    let mut locs = src.locations.clone();
    locs.sort();
    assert_eq!(
        locs,
        vec![
            ("alpha".to_string(), "hf".to_string()),
            ("beta".into(), "aaa_auto".into()),
            ("beta".into(), "zzz_hub".into())
        ]
    );
    w.stop().await;
}

#[tokio::test]
async fn test_an_export_of_a_subdirectory_maps_the_same_way() {
    let t = tempfile::tempdir().unwrap();
    let a = alpha(t.path());
    let b = node(
        t.path(),
        "beta",
        NodeOpts {
            roots: vec![RootCfg::new("shared", "hf-home").ro().at("/mnt/hfshare")],
            addresses: vec!["10.0.0.2".into()],
            hostname: Some("beta".into()),
            scan_disk: false,
            mounts: Some(vec![
                mount_dict("/", MountOpts::default()),
                nfs_mount("10.0.0.1", "/data/hf", "/mnt/hfshare", &["shared"]),
            ]),
            ..NodeOpts::default()
        },
    );
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w = world(t.path(), vec![a, b]).await;
    let c = w.library.models(None, None, None).unwrap()[0]
        .copies
        .clone();
    assert!(c.len() == 1 && c[0].visible_on == ["alpha", "beta"]);
    w.stop().await;
}

#[tokio::test]
async fn test_when_the_owner_is_not_a_configured_host_the_copy_stays_with_the_address() {
    let t = tempfile::tempdir().unwrap();
    let a = alpha(t.path());
    let b = nfs_beta(
        t.path(),
        Some(nfs_mount(
            "10.9.9.9",
            "/data",
            "/mnt/alpha",
            &["shared", "shared_g"],
        )),
    );
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w = world(t.path(), vec![a, b]).await;
    let copies = w.library.models(None, None, None).unwrap()[0]
        .copies
        .clone();
    assert_eq!(
        copies.iter().map(|c| c.host_id.clone()).collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    assert!(copies.iter().all(|c| c.visible_on == [c.host_id.clone()]));
    let d = copies[1].drive.clone().unwrap();
    assert!(d.remote && d.remote_host.as_deref() == Some("10.9.9.9") && d.remote_host_id.is_none());
    assert!(d.label == "NFS from 10.9.9.9" && d.remote_path.as_deref() == Some("/data"));
    w.stop().await;
}

#[tokio::test]
async fn test_a_configured_owner_that_does_not_list_the_copy_keeps_the_borrowers_copy() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (alpha(t.path()), beta(t.path()));
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w = world(t.path(), vec![a, b]).await;
    let copies = w.library.models(None, None, None).unwrap()[0]
        .copies
        .clone();
    assert!(copies.len() == 1 && copies[0].host_id == "beta" && copies[0].visible_on == ["beta"]);
    let d = copies[0].drive.clone().unwrap();
    assert!(d.remote_host_id.as_deref() == Some("alpha") && d.label == "NFS from Alpha");
    w.stop().await;
}

#[tokio::test]
async fn test_an_export_that_does_not_cover_the_copy_is_not_merged() {
    let t = tempfile::tempdir().unwrap();
    let a = alpha(t.path());
    let b = nfs_beta(
        t.path(),
        Some(nfs_mount("10.0.0.1", "/other", "/mnt/alpha", &["shared"])),
    );
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w = world(t.path(), vec![a, b]).await;
    let hosts: Vec<String> = w.library.models(None, None, None).unwrap()[0]
        .copies
        .iter()
        .map(|c| c.host_id.clone())
        .collect();
    assert_eq!(hosts, ["alpha", "beta"]);
    w.stop().await;
}

#[tokio::test]
async fn test_an_address_two_machines_report_identifies_nobody() {
    let t = tempfile::tempdir().unwrap();
    let a = nfs_alpha(t.path(), Some(vec!["10.0.0.1", "172.17.0.1"]));
    let g = node(
        t.path(),
        "gamma",
        NodeOpts {
            roots: vec![RootCfg::new("hf", "hf-home")],
            addresses: vec!["10.0.0.3".into(), "172.17.0.1".into()],
            scan_disk: false,
            ..NodeOpts::default()
        },
    );
    let b = nfs_beta(
        t.path(),
        Some(nfs_mount("172.17.0.1", "/data", "/mnt/alpha", &["shared"])),
    );
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w = world(t.path(), vec![a, g, b]).await;
    let copies = w.library.models(None, None, None).unwrap()[0]
        .copies
        .clone();
    assert_eq!(
        copies.iter().map(|c| c.host_id.clone()).collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    assert!(copies[1].drive.as_ref().unwrap().remote_host_id.is_none());
    w.stop().await;
}

#[tokio::test]
async fn test_loopback_nfs_and_non_nfs_remotes_are_never_merged() {
    let t = tempfile::tempdir().unwrap();
    let a = alpha(t.path());
    let b = nfs_beta(
        t.path(),
        Some(nfs_mount("127.0.0.1", "/data", "/mnt/alpha", &["shared"])),
    );
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w = world(t.path(), vec![a, b]).await;
    assert_eq!(
        w.library.models(None, None, None).unwrap()[0].copies.len(),
        2
    );
    w.stop().await;
    let x = t.path().join("x");
    let cifs = mount_dict(
        "/mnt/alpha",
        MountOpts {
            fstype: Some("cifs".into()),
            source: Some("//10.0.0.1/data".into()),
            model: None,
            remote: true,
            remote_host: Some("10.0.0.1".into()),
            remote_path: Some("/data".into()),
            roots: vec!["shared".into()],
            ..MountOpts::default()
        },
    );
    let b2 = nfs_beta(&x, Some(cifs));
    let a2 = alpha(&x);
    a2.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b2.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w2 = world(&x, vec![a2, b2]).await;
    let copies = w2.library.models(None, None, None).unwrap()[0]
        .copies
        .clone();
    assert!(copies.len() == 2 && copies[1].drive.as_ref().unwrap().label == "CIFS from Alpha");
    w2.stop().await;
}

#[tokio::test]
async fn test_the_owner_may_be_named_by_hostname() {
    let t = tempfile::tempdir().unwrap();
    for server in ["alpha.lan", "ALPHA", "alpha"] {
        let d = t.path().join(server);
        let a = alpha(&d);
        let b = nfs_beta(
            &d,
            Some(nfs_mount(server, "/data", "/mnt/alpha", &["shared"])),
        );
        a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
        b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
        let w = world(&d, vec![a, b]).await;
        assert_eq!(
            w.library.models(None, None, None).unwrap()[0].copies.len(),
            1,
            "{server}"
        );
        w.stop().await;
    }
}

#[tokio::test]
async fn test_the_merge_survives_a_restart_with_the_owner_down() {
    let t = tempfile::tempdir().unwrap();
    let db = t.path().join("console.db");
    let (a, b) = (alpha(t.path()), beta(t.path()));
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w1 = World::new(
        t.path(),
        vec![a, b],
        WorldOpts {
            store: Some(Arc::new(Store::open(&db).unwrap())),
            ..WorldOpts::default()
        },
    )
    .start()
    .await;
    w1.stop().await;
    let (a2, b2) = (alpha(t.path()), beta(t.path()));
    b2.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    a2.st().state = "down".into();
    a2.forget();
    let w2 = World::new(
        t.path(),
        vec![a2, b2],
        WorldOpts {
            store: Some(Arc::new(Store::open(&db).unwrap())),
            ..WorldOpts::default()
        },
    )
    .start()
    .await;
    let c = w2.library.models(None, None, None).unwrap()[0]
        .copies
        .clone();
    assert!(c.len() == 1 && c[0].visible_on == ["alpha", "beta"] && !c[0].deletable);
    w2.stop().await;
}

#[tokio::test]
async fn test_a_change_of_mounts_rebuilds_the_merge_without_a_new_scan() {
    let t = tempfile::tempdir().unwrap();
    let a = alpha(t.path());
    let b = nfs_beta(
        t.path(),
        Some(nfs_mount("10.9.9.9", "/data", "/mnt/alpha", &["shared"])),
    );
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    let w = world(t.path(), vec![a, b.clone()]).await;
    assert_eq!(
        w.library.models(None, None, None).unwrap()[0].copies.len(),
        2
    );
    b.set_mounts(vec![
        mount_with_roots("/", &["own"]),
        nfs_mount("10.0.0.1", "/data", "/mnt/alpha", &["shared"]),
    ]);
    w.bus.publish(
        "node.disks",
        json!({"host_id": "beta", "msg": Value::Object(b.disks())}),
    );
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(
        w.library.models(None, None, None).unwrap()[0].copies.len(),
        1
    );
    w.stop().await;
}

// ---- roots ------------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_roots_report_the_mount_the_owner_and_who_can_read_them() {
    let t = tempfile::tempdir().unwrap();
    let w = world(t.path(), vec![alpha(t.path()), beta(t.path())]).await;
    let shared = w.library.root("beta", "shared").unwrap();
    assert!(
        shared.shared_from.as_deref() == Some("alpha") && shared.visible_on == ["beta", "alpha"]
    );
    let m = shared.mount.clone().unwrap();
    assert!(m.label == "NFS from Alpha" && m.remote_host_id.as_deref() == Some("alpha"));
    let own = w.library.root("beta", "own").unwrap();
    assert!(
        own.shared_from.is_none() && own.visible_on == ["beta"] && own.mount.unwrap().mount == "/"
    );
    let owner_root = w.library.root("alpha", "hf").unwrap();
    assert!(owner_root.shared_from.is_none() && owner_root.visible_on == ["alpha", "beta"]);
    assert!(w.library.root("alpha", "nope").is_none() && w.library.roots("nobody").is_empty());
    w.stop().await;
}

#[tokio::test]
async fn test_hf_root_is_the_first_hf_root_and_auto_roots_count_when_their_scan_found_one() {
    let t = tempfile::tempdir().unwrap();
    let a = node(
        t.path(),
        "alpha",
        NodeOpts {
            scan_disk: false,
            roots: vec![
                RootCfg::new("g", "gguf"),
                RootCfg::new("auto1", "auto"),
                RootCfg::new("h", "hf-hub"),
                RootCfg::new("ol", "ollama"),
            ],
            ..NodeOpts::default()
        },
    );
    let b = node(
        t.path(),
        "beta",
        NodeOpts {
            scan_disk: false,
            roots: vec![RootCfg::new("auto2", "auto"), RootCfg::new("g2", "gguf")],
            ..NodeOpts::default()
        },
    );
    a.st().entries = vec![entry(
        "auto1",
        "hub/models--o--n",
        "hf",
        Some("o/n"),
        json!({}),
    )];
    b.st().entries = vec![gguf_entry("auto2", "x.gguf", json!({}))];
    let w = world(t.path(), vec![a, b]).await;
    use ec_hub::services::libport::LibraryPort as _;
    assert_eq!(w.library.hf_root("alpha").unwrap().id, "auto1");
    assert!(w.library.hf_root("beta").is_none());
    assert_eq!(w.library.ollama_root("alpha").unwrap().id, "ol");
    assert!(w.library.ollama_root("beta").is_none() && w.library.hf_root("nobody").is_none());
    let u = w.library.root_usage("alpha", "auto1");
    assert!(u.hf_home && !u.hf_hub && u.entries == 1);
    w.stop().await;
}

#[tokio::test]
async fn test_a_mount_the_node_claims_for_a_root_wins_over_the_longest_prefix() {
    let t = tempfile::tempdir().unwrap();
    let b = node(
        t.path(),
        "beta",
        NodeOpts {
            scan_disk: false,
            roots: vec![RootCfg::new("r", "hf-home").at("/data/hf")],
            mounts: Some(vec![
                mount_with_roots("/", &[]),
                mount_dict(
                    "/data",
                    MountOpts {
                        fstype: Some("zfs".into()),
                        source: Some("tank/models".into()),
                        model: None,
                        roots: vec!["r".into()],
                        ..MountOpts::default()
                    },
                ),
                mount_with_roots("/data/hf", &[]),
            ]),
            ..NodeOpts::default()
        },
    );
    let w = world(t.path(), vec![b]).await;
    let m = w.library.root("beta", "r").unwrap().mount.unwrap();
    assert!(m.mount == "/data" && m.label == "tank (zfs)");
    w.stop().await;
}

#[tokio::test]
async fn test_drive_labels() {
    let remote = |fstype: &str, path: &str| MountOpts {
        fstype: Some(fstype.into()),
        model: None,
        remote: true,
        remote_host: Some("nas.lan".into()),
        remote_path: Some(path.into()),
        ..MountOpts::default()
    };
    let cases: Vec<(&str, MountOpts, &str)> = vec![
        (
            "/",
            MountOpts {
                model: Some("Samsung 990".into()),
                ..MountOpts::default()
            },
            "/ (ext4, Samsung 990)",
        ),
        (
            "/",
            MountOpts {
                model: None,
                ..MountOpts::default()
            },
            "/ (ext4)",
        ),
        (
            "/data",
            MountOpts {
                fstype: Some("zfs".into()),
                source: Some("tank/models".into()),
                model: None,
                ..MountOpts::default()
            },
            "tank (zfs)",
        ),
        (
            "/data",
            MountOpts {
                fstype: Some("zfs".into()),
                source: Some("tank".into()),
                model: None,
                ..MountOpts::default()
            },
            "tank (zfs)",
        ),
        ("/mnt/x", remote("nfs4", "/x"), "NFS from nas.lan"),
        ("/mnt/x", remote("cifs", "share"), "CIFS from nas.lan"),
        (
            "/mnt/x",
            MountOpts {
                fstype: None,
                model: None,
                ..MountOpts::default()
            },
            "/mnt/x",
        ),
    ];
    for (i, (mnt, o, label)) in cases.into_iter().enumerate() {
        let t = tempfile::tempdir().unwrap();
        let path = format!("{}/hf", mnt.trim_end_matches('/'));
        let a = node(
            t.path(),
            "alpha",
            NodeOpts {
                scan_disk: false,
                roots: vec![RootCfg::new("r", "hf-home").at(path)],
                mounts: Some(vec![mount_dict(mnt, o)]),
                ..NodeOpts::default()
            },
        );
        a.st().entries = vec![hf_entry("r", "org/m", json!({}))];
        let w = world(t.path(), vec![a]).await;
        let d = w.library.models(None, None, None).unwrap()[0].copies[0]
            .drive
            .clone()
            .unwrap();
        assert_eq!(d.label, label, "case {i}");
        w.stop().await;
    }
}

#[tokio::test]
async fn test_other_entries_do_not_break_the_merge() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (alpha(t.path()), beta(t.path()));
    a.st().entries = vec![entry("hf", "weird", "other", Some("weird"), json!({}))];
    b.st().entries = vec![entry("shared", "weird", "other", Some("weird"), json!({}))];
    let w = world(t.path(), vec![a, b]).await;
    assert_eq!(
        w.library.models(None, None, None).unwrap()[0].copies[0].visible_on,
        ["alpha", "beta"]
    );
    w.stop().await;
}
