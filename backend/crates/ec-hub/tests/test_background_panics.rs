//! A panic inside a background job never leaves a row stuck or the service dead (audit lessons, PORTING.md §A12).
//! Python wraps `DownloadService._run` in `except Exception` ("a worker must never die silently"): the
//! row fails with `download_failed` and the next download runs. In Rust a panic is that unexpected exception; the run
//! catches it, marks the row failed with a redacted message, and the service goes on.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/wsc_data.rs"]
mod wsc_data;

use std::sync::Arc;

use ec_hub::domain::models::DownloadRequest;
use serde_json::json;
use wsc_data::*;

fn req() -> DownloadRequest {
    DownloadRequest {
        repo_id: "org/m".into(),
        revision: None,
        allow_patterns: None,
        host_id: "spark".into(),
        root_id: Some("hf".into()),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_panicking_download_fails_its_row_and_the_next_download_runs() {
    ec_hub::panics::install_panic_hook();
    let tmp = tempfile::tempdir().unwrap();
    let nd = node(
        tmp.path(),
        "spark",
        NodeOpts {
            roots: vec![RootCfg::new("hf", "hf-home")],
            ..NodeOpts::default()
        },
    );
    let w = World::new(tmp.path(), vec![Arc::clone(&nd)], WorldOpts::default());
    w.hub
        .add_files("org/m", &files(), Some(json!({"model_type": "x"})));
    let w = w.start().await;

    // the node's fs.stat panics (a bug in the run, as Python's unexpected exception); every download stats first
    nd.on(
        "fs.stat",
        Arc::new(|_, _| panic!("a bug in the run, token=hf_supersecrettokenvalue123")),
    );

    let d = w.downloads.create(&req()).await.unwrap();
    w.downloads.wait(&d.id).await;
    let failed = w.downloads.get(&d.id).unwrap();
    assert_eq!(failed.state, "failed", "{failed:?}");
    assert_eq!(failed.error_code.as_deref(), Some("download_failed"));
    let err = failed.error.clone().unwrap_or_default();
    assert!(err.starts_with("panic: "), "{err}");
    assert!(!err.contains("hf_supersecrettokenvalue123"), "{err}"); // redacted before it is stored

    // the service is alive: the same download, without the bug, completes
    nd.st().overrides.remove("fs.stat");
    let d2 = w.downloads.create(&req()).await.unwrap();
    w.downloads.wait(&d2.id).await;
    let done = w.downloads.get(&d2.id).unwrap();
    assert_eq!(done.state, "completed", "{done:?}");
}
