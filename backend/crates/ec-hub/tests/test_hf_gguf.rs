//! Port of backend/tests/data/test_hf_gguf.py: GGUF quantisation grouping (the download picker's data) and the HF
//! client's mid-download failure mapping. Python mocks the transport with httpx.MockTransport; here the
//! client talks TLS to a recording `openssl s_server` (tests/common/wsc_tls.rs) through a CONNECT proxy.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

#[path = "common/wsb_egress.rs"]
mod wsb_egress;
#[path = "common/wsc_data.rs"]
mod wsc_data;
#[path = "common/wsc_tls.rs"]
mod wsc_tls;

use std::sync::Arc;
use std::time::Duration;

use ec_hub::domain::errors::AppError;
use ec_hub::domain::ports::{HubClient, RepoFile};
use ec_hub::services::downloads::path_matches;
use ec_hub::services::hf::{HfService, gguf_quant, gguf_stem, group_gguf};
use ec_hub::services::hf_http::{EgressConfig, HfHttpClient};
use serde_json::json;
use wsb_egress::{ConnectProxy, has_openssl, make_certs};
use wsc_tls::{TlsRecorder, response};

const MIB: usize = 1024 * 1024;
const COMMIT_SHA: &str = "c0ffee00000000000000000000000000000000ff";

/// The 28 parametrized cases of `test_gguf_quant_token_from_a_repo_path`.
const QUANTS: [(&str, &str); 28] = [
    ("Qwen3-8B-Q4_K_M.gguf", "Q4_K_M"),
    ("Qwen3-8B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
    ("Qwen3-8B-UD-IQ2_XXS.gguf", "UD-IQ2_XXS"),
    ("model-IQ2_XXS.gguf", "IQ2_XXS"),
    ("model-IQ4_NL.gguf", "IQ4_NL"),
    ("model-BF16.gguf", "BF16"),
    ("model-F16.gguf", "F16"),
    ("model-F32.gguf", "F32"),
    ("gpt-oss-120b-MXFP4.gguf", "MXFP4"),
    ("gpt-oss-120b-MXFP4_MOE.gguf", "MXFP4_MOE"),
    ("model-Q8_0.gguf", "Q8_0"),
    ("model-Q5_K_S.gguf", "Q5_K_S"),
    ("model-Q2_K.gguf", "Q2_K"),
    ("model-Q3_K_L.gguf", "Q3_K_L"),
    ("model-TQ1_0.gguf", "TQ1_0"),
    ("qwen2.5-7b-instruct-q4_k_m.gguf", "Q4_K_M"),
    ("model.Q6_K.gguf", "Q6_K"),
    ("model.i1-Q4_K_M.gguf", "i1-Q4_K_M"),
    ("model-Q4_K_M-00001-of-00003.gguf", "Q4_K_M"),
    ("Q4_K_M/Model-Q4_K_M-00002-of-00003.gguf", "Q4_K_M"),
    ("BF16/Big-00001-of-00010.gguf", "BF16"),
    ("mmproj-F16.gguf", "mmproj-F16"),
    ("mmproj-BF16.gguf", "mmproj-BF16"),
    ("model.gguf", "model"),
    ("sub/ggml-model.gguf", "ggml-model"),
    ("mmproj.gguf", "mmproj"),
    ("Llama-3.1-8B-Instruct-Q4_K_M.GGUF", "Q4_K_M"),
    ("my-F16x-model.gguf", "my-F16x-model"),
];

#[test]
fn test_gguf_quant_token_from_a_repo_path() {
    for (path, quant) in QUANTS {
        assert_eq!(gguf_quant(path), quant, "{path}");
    }
}

#[test]
fn test_gguf_stem_drops_the_extension_and_the_split_suffix() {
    assert_eq!(gguf_stem("a/b-Q4_K_M-00001-of-00003.gguf"), "a/b-Q4_K_M");
    assert_eq!(gguf_stem("c.GGUF"), "c");
    assert_eq!(gguf_stem("d"), "d");
}

#[test]
fn test_groups_keep_split_sets_together_and_sort_smallest_first() {
    let files = [
        ("README.md", 10),
        ("config.json", 5),
        ("m-Q8_0.gguf", 800),
        ("m-Q4_K_M.gguf", 400),
        ("m-BF16-00001-of-00002.gguf", 700),
        ("m-BF16-00002-of-00002.gguf", 650),
        ("m-UD-Q4_K_XL.gguf", 450),
        ("mmproj-F16.gguf", 50),
    ];
    let groups = group_gguf(files);
    assert_eq!(
        groups
            .iter()
            .map(|g| (g.quant.as_str(), g.total_bytes))
            .collect::<Vec<_>>(),
        [
            ("mmproj-F16", 50),
            ("Q4_K_M", 400),
            ("UD-Q4_K_XL", 450),
            ("Q8_0", 800),
            ("BF16", 1350)
        ]
    );
    let bf16 = groups.iter().find(|g| g.quant == "BF16").unwrap();
    assert_eq!(
        bf16.files,
        ["m-BF16-00001-of-00002.gguf", "m-BF16-00002-of-00002.gguf"]
    );
    // only .gguf files
    assert!(
        !groups
            .iter()
            .flat_map(|g| &g.files)
            .any(|f| f.contains("README") || f.ends_with(".json"))
    );
    assert!(group_gguf([("a.safetensors", 1)]).is_empty());
}

#[test]
fn test_the_files_of_a_group_are_usable_as_allow_patterns() {
    let groups = group_gguf([
        ("dir/m-Q4_K_M-00001-of-00002.gguf", 1),
        ("dir/m-Q4_K_M-00002-of-00002.gguf", 1),
        ("dir/m-Q8_0.gguf", 1),
    ]);
    let g = groups.iter().find(|x| x.quant == "Q4_K_M").unwrap();
    let kept: Vec<&str> = [
        "dir/m-Q4_K_M-00001-of-00002.gguf",
        "dir/m-Q4_K_M-00002-of-00002.gguf",
        "dir/m-Q8_0.gguf",
        "README.md",
    ]
    .into_iter()
    .filter(|p| path_matches(p, Some(&g.files)))
    .collect();
    assert_eq!(kept, g.files);
    // an exact path wins over fnmatch's reading of [..]
    assert!(path_matches("a[1].gguf", Some(&["a[1].gguf".to_string()])));
}

#[test]
fn test_hf_service_gguf_groups_accepts_repo_files_and_dicts() {
    let as_files = HfService::gguf_groups(&[
        RepoFile {
            path: "m-Q4_K_M.gguf".into(),
            size: 4,
            sha256: None,
        },
        RepoFile {
            path: "x.txt".into(),
            size: 1,
            sha256: None,
        },
    ]);
    let as_dicts = HfService::gguf_groups_of_dicts(&[
        json!({"path": "m-Q4_K_M.gguf", "size": 4, "sha256": null}),
        json!({"path": "bad", "size": "x"}),
        json!({"size": 1}),
    ]);
    assert_eq!(as_files, as_dicts);
    assert!(as_files[0].quant == "Q4_K_M" && as_files[0].total_bytes == 4);
}

#[tokio::test]
async fn test_model_detail_carries_gguf_groups() {
    let hub = Arc::new(wsc_data::DataHub::new());
    hub.add(
        "org/m-GGUF",
        &[
            ("m-Q4_K_M.gguf", b"1234"),
            ("m-Q8_0.gguf", b"12345678"),
            ("README.md", b"x"),
        ],
    );
    hub.add_full(
        "org/plain",
        &[("model.safetensors", b"1234"), ("config.json", b"{}")],
        wsc_data::COMMIT,
        Some(json!({"model_type": "x"})),
    );
    let svc = HfService::with_ttl(hub, Duration::ZERO);
    let detail = svc.detail("org/m-GGUF", None).await.unwrap();
    assert_eq!(
        detail
            .gguf_groups
            .iter()
            .map(|g| (g.quant.as_str(), g.files.clone(), g.total_bytes))
            .collect::<Vec<_>>(),
        [
            ("Q4_K_M", vec!["m-Q4_K_M.gguf".to_string()], 4),
            ("Q8_0", vec!["m-Q8_0.gguf".to_string()], 8)
        ]
    );
    let dumped = serde_json::to_value(&detail).unwrap();
    assert_eq!(
        dumped["gguf_groups"][0],
        json!({"quant": "Q4_K_M", "files": ["m-Q4_K_M.gguf"], "total_bytes": 4})
    );
    assert!(
        svc.detail("org/plain", None)
            .await
            .unwrap()
            .gguf_groups
            .is_empty()
    );
}

// ---- a connection that dies in the middle of a file -----------------------------------------------------------------
struct Upstream {
    _dir: tempfile::TempDir,
    hub: TlsRecorder,
    client: HfHttpClient,
    _proxy: ConnectProxy,
}

async fn upstream(token: Option<&'static str>) -> Option<Upstream> {
    if !has_openssl() {
        eprintln!("SKIP: no openssl CLI");
        return None;
    }
    let dir = tempfile::tempdir().unwrap();
    let certs = make_certs(&dir.path().join("pki"));
    let hub = TlsRecorder::start(&certs);
    let proxy = ConnectProxy::start().await;
    let client = HfHttpClient::new(
        &hub.endpoint(),
        vec!["127.0.0.1".into()],
        Arc::new(move || token.map(str::to_string)),
        EgressConfig {
            proxy: Some(proxy.url()),
            ca_bundle: Some(certs.ca.clone()),
            require_proxy: true,
        },
    );
    Some(Upstream {
        _dir: dir,
        hub,
        client,
        _proxy: proxy,
    })
}

/// Python parametrizes the transport's error (ReadError, RemoteProtocolError, ReadTimeout); every one reaches the
/// same mapping. Here the server sends 1 MiB of a promised 2 MiB and the connection dies (a 120 s read timeout is not
/// waited for: it takes the same path).
#[tokio::test]
async fn test_a_connection_that_drops_mid_file_is_an_upstream_error_to_resume() {
    let Some(u) = upstream(None).await else {
        return;
    };
    let mut head = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\n\r\n",
        2 * MIB
    );
    head.push_str(&"a".repeat(MIB));
    u.hub
        .route_then_drop(&format!("/org/m/resolve/{COMMIT_SHA}/w.safetensors"), head);
    let mut stream = u
        .client
        .open_file("org/m", COMMIT_SHA, "w.safetensors", 0)
        .await
        .unwrap();
    let mut got: Vec<usize> = Vec::new();
    let err = loop {
        match stream.chunk().await {
            Ok(Some(c)) => got.push(c.len()),
            Ok(None) => panic!("the stream ended cleanly after {got:?}"),
            Err(e) => break e,
        }
    };
    assert_eq!(got, [MIB]);
    let AppError::Problem(p) = &err else {
        panic!("{err:?}")
    };
    assert!(
        p.code == "upstream_error" && p.detail.contains("resume"),
        "{p:?}"
    );
    // no library exception name as the headline
    let headline = p.detail.split('(').next().unwrap_or("");
    assert!(
        ![
            "ReadError",
            "RemoteProtocolError",
            "ReadTimeout",
            "hyper",
            "reqwest"
        ]
        .iter()
        .any(|n| headline.contains(n)),
        "{headline}"
    );
    stream.close().await;
}

#[tokio::test]
async fn test_the_range_header_and_token_are_sent_to_hf_only() {
    let Some(u) = upstream(Some("hf_secrettoken123")).await else {
        return;
    };
    u.hub.route(
        &format!("/org/m/resolve/{COMMIT_SHA}/dir/w.safetensors"),
        response(206, &[("content-range", "bytes 7-9/10")], "xyz"),
    );
    let mut s = u
        .client
        .open_file("org/m", COMMIT_SHA, "dir/w.safetensors", 7)
        .await
        .unwrap();
    assert!(s.offset() == 7 && s.total() == Some(10));
    let reqs = u.hub.requests();
    let first = reqs.first().unwrap();
    assert_eq!(first.header("range"), Some("bytes=7-"));
    assert_eq!(
        first.header("host"),
        Some(format!("127.0.0.1:{}", u.hub.port).as_str())
    );
    assert_eq!(
        first.header("authorization"),
        Some("Bearer hf_secrettoken123")
    );
    assert!(!first.target.contains("hf_secrettoken123"));
    s.close().await;
}
