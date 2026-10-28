//! The library and data rows of backend/tests/core/test_units.py: hub parsing (build_model_info, detect_quantization),
//! the egress allowlist and the HF HTTP client. Python mocks the hub with respx; here the client talks TLS to a
//! recording `openssl s_server` (tests/common/wsc_tls.rs) through a CONNECT proxy, with a throwaway CA as the egress CA
//! bundle, all on 127.0.0.1.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

#[path = "common/kit.rs"]
mod kit;
#[path = "common/wsb_egress.rs"]
mod wsb_egress;
#[path = "common/wsc_tls.rs"]
mod wsc_tls;

use std::sync::Arc;

use ec_hub::domain::errors::AppError;
use ec_hub::domain::ports::{HubClient, HubModel, RepoFile};
use ec_hub::services::hf::{build_model_info, detect_quantization};
use ec_hub::services::hf_http::{EgressConfig, HfHttpClient, host_allowed};
use kit::fakefleet::qwen_cfg;
use serde_json::{Map, Value, json};
use wsb_egress::{Certs, ConnectProxy, has_openssl, make_certs};
use wsc_tls::{TlsRecorder, response};

fn hub(cfg: Option<Value>, files: Vec<RepoFile>) -> HubModel {
    HubModel {
        repo_id: "a/b".into(),
        sha: Some("s".into()),
        gated: false,
        license: None,
        pipeline_tag: None,
        library_name: None,
        tags: Vec::new(),
        files,
        safetensors_params: Vec::new(),
        safetensors_total: None,
        config: cfg.and_then(|c| c.as_object().cloned()),
        downloads: None,
        likes: None,
    }
}

fn file(path: &str, size: i64) -> RepoFile {
    RepoFile {
        path: path.into(),
        size,
        sha256: None,
    }
}

fn obj(v: &Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

// ---- hub parsing ----------------------------------------------------------------------------------------------------
#[test]
fn test_build_model_info_dense_qwen() {
    let mut h = hub(
        Some(qwen_cfg()),
        vec![
            file("a.safetensors", 10),
            file("b.safetensors", 5),
            file("x.md", 999),
        ],
    );
    h.safetensors_total = Some(32_000_000_000);
    let m = build_model_info(&h, None).unwrap();
    assert_eq!(
        (m.num_layers, m.num_kv_heads, m.head_dim, m.hidden_size),
        (Some(64), Some(8), Some(128), Some(5120))
    );
    assert!(m.weight_bytes == Some(15) && m.num_params == Some(32_000_000_000) && !m.is_moe);
    assert_eq!(m.dtype.as_deref(), Some("bfloat16"));
}

#[test]
fn test_build_model_info_moe_mla_nested_text_config_and_missing() {
    let moe = build_model_info(
        &hub(
            Some(json!({"num_experts": 128, "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4})),
            vec![],
        ),
        None,
    )
    .unwrap();
    // never invents kv heads
    assert!(moe.is_moe && moe.head_dim == Some(16) && moe.num_kv_heads.is_none());
    let mla = build_model_info(
        &hub(
            Some(json!({"kv_lora_rank": 512, "hidden_size": 64, "num_attention_heads": 4, "n_routed_experts": 8})),
            vec![],
        ),
        None,
    )
    .unwrap();
    assert!(mla.kv_lora_rank == Some(512) && mla.head_dim.is_none() && mla.is_moe);
    let nested = build_model_info(
        &hub(
            Some(json!({"architectures": ["VL"], "text_config": {"num_hidden_layers": 7, "hidden_size": 8,
                         "num_attention_heads": 2, "num_key_value_heads": 1}})),
            vec![],
        ),
        None,
    )
    .unwrap();
    assert!(nested.num_layers == Some(7) && nested.num_kv_heads == Some(1));
    assert_eq!(nested.architectures, ["VL"]);
    let empty = build_model_info(&hub(None, vec![]), None).unwrap();
    assert!(empty.num_layers.is_none() && empty.raw_config.is_empty());
}

#[test]
fn test_weight_bytes_fall_back_to_safetensors_metadata() {
    let mut h = hub(Some(json!({})), vec![]);
    h.safetensors_params = vec![("BF16".into(), 10), ("F8_E4M3".into(), 4)];
    assert_eq!(build_model_info(&h, None).unwrap().weight_bytes, Some(24));
}

/// The eight parametrized cases of `test_detect_quantization` (cfg0..cfg7).
#[test]
fn test_detect_quantization() {
    let cases: [(Value, Option<&str>); 8] = [
        (
            json!({"quantization_config": {"quant_method": "fp8"}}),
            Some("fp8"),
        ),
        (
            json!({"quantization_config": {"quant_method": "mxfp4"}}),
            Some("mxfp4"),
        ),
        (
            json!({"quantization_config": {"quant_method": "compressed-tensors",
                   "config_groups": {"g": {"weights": {"num_bits": 4, "type": "float"}}}}}),
            Some("nvfp4"),
        ),
        (
            json!({"quantization_config": {"quant_method": "compressed-tensors",
                   "config_groups": {"g": {"weights": {"num_bits": 8, "type": "float"}}}}}),
            Some("fp8"),
        ),
        (
            json!({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}),
            Some("nvfp4"),
        ),
        (
            json!({"quantization_config": {"quant_method": "awq"}}),
            Some("awq"),
        ),
        (
            json!({"quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": true}}),
            Some("bnb4"),
        ),
        (json!({}), None),
    ];
    for (i, (cfg, want)) in cases.iter().enumerate() {
        assert_eq!(
            detect_quantization(&obj(cfg), "", &[]).as_deref(),
            *want,
            "cfg{i}"
        );
    }
}

#[test]
fn test_detect_quantization_from_name_and_tags() {
    assert_eq!(
        detect_quantization(&Map::new(), "nvidia/Llama-NVFP4", &[]).as_deref(),
        Some("nvfp4")
    );
    assert_eq!(
        detect_quantization(&Map::new(), "x/y", &["gptq".to_string()]).as_deref(),
        Some("gptq")
    );
}

// ---- egress allowlist (ADR 8) ---------------------------------------------------------------------------------------
/// The seven parametrized cases of `test_host_allowed`.
#[test]
fn test_host_allowed() {
    let pats: Vec<String> = [
        "huggingface.co",
        "cdn-lfs*.huggingface.co",
        "cas-bridge.xethub.hf.co",
        "*.hf.co",
    ]
    .map(String::from)
    .to_vec();
    for (host, ok) in [
        ("huggingface.co", true),
        ("cdn-lfs-us-1.huggingface.co", true),
        ("cas-bridge.xethub.hf.co", true),
        ("x.hf.co", true),
        ("evil.com", false),
        ("huggingface.co.evil.com", false),
        ("nothf.co", false),
    ] {
        assert_eq!(host_allowed(host, &pats), ok, "{host}");
    }
}

// ---- the HF HTTP client ---------------------------------------------------------------------------------------------
struct Upstream {
    _dir: tempfile::TempDir,
    certs: Certs,
    proxy: ConnectProxy,
    hub: TlsRecorder,
}

async fn upstream() -> Option<Upstream> {
    if !has_openssl() {
        eprintln!("SKIP: no openssl CLI");
        return None;
    }
    let dir = tempfile::tempdir().unwrap();
    let certs = make_certs(&dir.path().join("pki"));
    let hub = TlsRecorder::start(&certs);
    Some(Upstream {
        _dir: dir,
        proxy: ConnectProxy::start().await,
        hub,
        certs,
    })
}

fn client(u: &Upstream, token: Option<&'static str>) -> HfHttpClient {
    HfHttpClient::new(
        &u.hub.endpoint(),
        vec!["127.0.0.1".into()],
        Arc::new(move || token.map(str::to_string)),
        EgressConfig {
            proxy: Some(u.proxy.url()),
            ca_bundle: Some(u.certs.ca.clone()),
            require_proxy: true,
        },
    )
}

fn problem(e: &AppError) -> (u16, String, String) {
    match e {
        AppError::Problem(p) => (p.status, p.code.clone(), p.detail.clone()),
        AppError::Node(n) => (0, n.code.clone(), n.message.clone()),
    }
}

#[tokio::test]
async fn test_http_client_blocks_redirect_off_allowlist_and_never_puts_token_in_url() {
    let Some(u) = upstream().await else { return };
    let c = client(&u, Some("hf_secrettokenvalue"));
    u.hub.route(
        "/x/y/resolve/main/f",
        response(302, &[("location", "https://evil.example/f")], ""),
    );
    let e = c.open_file("x/y", "main", "f", 0).await.err().unwrap();
    let (status, code, detail) = problem(&e);
    assert!(status == 403 && code == "egress_denied", "{e:?}");
    assert!(!detail.contains("secrettoken"));
    // nothing went to the redirect target: the proxy only ever tunnelled to the hub
    assert!(
        u.proxy
            .targets()
            .iter()
            .all(|t| t.starts_with("127.0.0.1:")),
        "{:?}",
        u.proxy.targets()
    );
    u.hub.route(
        "/api/models",
        response(200, &[("content-type", "application/json")], "[]"),
    );
    let hits = c
        .search("q", None, None, None, "downloads", 3, true)
        .await
        .unwrap();
    assert!(hits.is_empty());
    let reqs = u.hub.requests();
    let req = reqs
        .iter()
        .rev()
        .find(|r| r.path() == "/api/models")
        .unwrap();
    assert!(!req.target.contains("secrettoken"), "{}", req.target);
    assert_eq!(
        req.header("authorization"),
        Some("Bearer hf_secrettokenvalue")
    );
}

#[tokio::test]
async fn test_http_client_maps_statuses_and_parses_model() {
    let Some(u) = upstream().await else { return };
    let c = client(&u, None);
    let model = json!({"id": "a/b", "sha": "abc", "gated": "manual", "tags": ["license:mit"],
                       "safetensors": {"total": 5, "parameters": {"BF16": 5}},
                       "siblings": [{"rfilename": "m.safetensors", "size": 10, "lfs": {"sha256": "ff", "size": 10}}]});
    u.hub.route(
        "/api/models/a/b/revision/main",
        response(
            200,
            &[("content-type", "application/json")],
            &model.to_string(),
        ),
    );
    u.hub
        .route("/a/b/resolve/abc/config.json", response(401, &[], ""));
    let m = c.model("a/b", None).await.unwrap();
    assert!(
        m.gated
            && m.license.as_deref() == Some("mit")
            && m.config.is_none()
            && m.sha.as_deref() == Some("abc")
    );
    assert_eq!(m.files[0].sha256.as_deref(), Some("ff"));
    // no Authorization header without a token
    assert!(
        u.hub
            .requests()
            .iter()
            .all(|r| r.header("authorization").is_none())
    );
    u.hub
        .route("/api/models/no/pe/revision/main", response(404, &[], ""));
    let e = c.model("no/pe", None).await.err().unwrap();
    assert_eq!(problem(&e).0, 404, "{e:?}");
    u.hub
        .route("/api/models/gt/ed/revision/main", response(403, &[], ""));
    let e = c.model("gt/ed", None).await.err().unwrap();
    let (status, code, detail) = problem(&e);
    assert!(
        status == 403 && code == "hf_gated" && detail.contains("HF_TOKEN"),
        "{e:?}"
    );
}
