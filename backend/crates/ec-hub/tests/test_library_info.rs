//! Port of backend/tests/data/test_library_info.py: model_info for library copies (hf and plain from config.json
//! through the node, gguf and ollama from the scanned header) and the pure GGUF/Ollama helpers.

#![allow(clippy::unwrap_used)]

#[path = "common/wsc_data.rs"]
mod wsc_data;

use std::path::Path;
use std::sync::Arc;

use ec_hub::domain::refs::ModelRef;
use ec_hub::services::gguf_info::{
    model_info_from_entry, normalize_quant, parse_size_label, quant_from_file_type, shard_names,
};
use ec_hub::services::libport::LibraryPort as _;
use serde_json::{Value, json};
use wsc_data::*;

fn qwen() -> Value {
    json!({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 5120, "num_hidden_layers": 64,
           "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 40960,
           "torch_dtype": "bfloat16"})
}

fn n(tmp: &Path) -> Arc<DataNode> {
    node(
        tmp,
        "alpha",
        NodeOpts {
            scan_disk: false,
            roots: vec![
                RootCfg::new("hf", "hf-home"),
                RootCfg::new("gguf", "gguf"),
                RootCfg::new("ro", "plain").ro(),
                RootCfg::new("ol", "ollama"),
            ],
            ..NodeOpts::default()
        },
    )
}

fn r(format: &str, root: &str, rel: &str) -> ModelRef {
    ModelRef {
        format: format.into(),
        repo_id: None,
        root_id: Some(root.into()),
        rel: Some(rel.into()),
        name: None,
    }
}

fn set_config(a: &DataNode, root: &str, rel: &str, cfg: Value) {
    a.st().configs.insert((root.into(), rel.into()), cfg);
}

async fn world(tmp: &Path, nodes: Vec<Arc<DataNode>>) -> World {
    World::new(tmp, nodes, WorldOpts::default()).start().await
}

#[tokio::test]
async fn test_hf_info_comes_from_the_nodes_config_and_the_copys_size() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path());
    a.st().entries = vec![hf_entry(
        "hf",
        "Qwen/Qwen3-32B",
        json!({"size": 65_000_000_000_i64, "params": 32_000_000_000_i64}),
    )];
    set_config(&a, "hf", "hub/models--Qwen--Qwen3-32B", qwen());
    let w = world(t.path(), vec![a.clone()]).await;
    let info = w
        .library
        .model_info("alpha", &ModelRef::hf("Qwen/Qwen3-32B"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (
            info.repo_id.as_str(),
            info.revision.as_deref(),
            info.num_layers,
            info.num_kv_heads,
            info.head_dim
        ),
        ("Qwen/Qwen3-32B", Some(COMMIT), Some(64), Some(8), Some(128))
    );
    assert!(
        info.max_position_embeddings == Some(40960) && info.model_type.as_deref() == Some("qwen3")
    );
    assert_eq!(info.dtype.as_deref(), Some("bfloat16"));
    assert!(info.weight_bytes == Some(65_000_000_000) && info.num_params == Some(32_000_000_000));
    assert_eq!(
        a.ops("models.config")
            .into_iter()
            .map(Value::Object)
            .collect::<Vec<_>>(),
        [json!({"root": "hf", "rel": "hub/models--Qwen--Qwen3-32B"})]
    );
    w.stop().await;
}

#[tokio::test]
async fn test_info_is_cached_until_the_library_changes() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path());
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    set_config(&a, "hf", "hub/models--org--m", qwen());
    let w = world(t.path(), vec![a.clone()]).await;
    let rf = ModelRef::hf("org/m");
    let mut first = w.library.model_info("alpha", &rf).await.unwrap().unwrap();
    first.num_layers = Some(1); // a caller scribbling on its copy ...
    let second = w.library.model_info("alpha", &rf).await.unwrap().unwrap();
    assert!(second.num_layers == Some(64) && a.ops("models.config").len() == 1); // ... changes nothing
    a.set_entries(vec![hf_entry("hf", "org/m", json!({"size": 5}))]);
    w.library.refresh("alpha").await.unwrap();
    assert_eq!(
        w.library
            .model_info("alpha", &rf)
            .await
            .unwrap()
            .unwrap()
            .weight_bytes,
        Some(5)
    );
    assert_eq!(a.ops("models.config").len(), 2);
    w.stop().await;
}

#[tokio::test]
async fn test_a_missing_config_gives_only_what_is_known() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path());
    a.st().entries = vec![hf_entry(
        "hf",
        "org/m",
        json!({"size": 777, "quant": "fp8"}),
    )];
    let w = world(t.path(), vec![a]).await;
    let info = w
        .library
        .model_info("alpha", &ModelRef::hf("org/m"))
        .await
        .unwrap()
        .unwrap();
    assert!(info.weight_bytes == Some(777) && info.quantization.as_deref() == Some("fp8"));
    assert!(info.num_layers.is_none() && info.num_kv_heads.is_none() && info.head_dim.is_none());
    w.stop().await;
}

#[tokio::test]
async fn test_plain_info_reads_the_directorys_config() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path());
    a.st().entries = vec![entry(
        "ro",
        "mydir",
        "plain",
        Some("mydir"),
        json!({"size": 4242}),
    )];
    let mut cfg = qwen();
    cfg["quantization_config"] = json!({"quant_method": "modelopt", "quant_algo": "FP8"});
    set_config(&a, "ro", "mydir", cfg);
    let w = world(t.path(), vec![a]).await;
    let info = w
        .library
        .model_info("alpha", &r("plain", "ro", "mydir"))
        .await
        .unwrap()
        .unwrap();
    assert!(
        info.repo_id == "mydir"
            && info.quantization.as_deref() == Some("fp8")
            && info.weight_bytes == Some(4242)
    );
    w.stop().await;
}

#[tokio::test]
async fn test_info_falls_back_to_a_machine_that_shares_the_storage_when_the_owner_is_down() {
    let t = tempfile::tempdir().unwrap();
    let (a, b) = (nfs_alpha(t.path(), None), nfs_beta(t.path(), None));
    a.st().entries = vec![hf_entry("hf", "org/m", json!({}))];
    b.st().entries = vec![hf_entry("shared", "org/m", json!({}))];
    set_config(&b, "shared", "hub/models--org--m", qwen());
    let w = world(t.path(), vec![a.clone(), b]).await;
    a.st().state = "down".into();
    let info = w
        .library
        .model_info("beta", &ModelRef::hf("org/m"))
        .await
        .unwrap()
        .unwrap();
    assert!(info.num_layers == Some(64) && a.ops("models.config").is_empty());
    w.stop().await;
}

#[tokio::test]
async fn test_gguf_info_from_the_nodes_header_facts() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path());
    a.st().entries = vec![gguf_entry(
        "gguf",
        "m/Qwen3-8B-Q4_K_M.gguf",
        json!({"name": "Qwen3-8B", "size": 5_000_000_000_i64, "params": 8_190_000_000_i64, "arch": "qwen3",
               "context": 40960, "shards": null,
               "meta": {"block_count": 36, "embedding_length": 4096, "head_count": 32, "head_count_kv": 8,
                        "expert_count": null, "file_type": 15, "complete": true}}),
    )];
    let w = world(t.path(), vec![a.clone()]).await;
    let info = w
        .library
        .model_info("alpha", &r("gguf", "gguf", "m/Qwen3-8B-Q4_K_M.gguf"))
        .await
        .unwrap()
        .unwrap();
    assert!(
        info.repo_id == "Qwen3-8B"
            && info.architectures == ["qwen3"]
            && info.model_type.as_deref() == Some("qwen3")
    );
    assert_eq!(
        (
            info.num_layers,
            info.num_attention_heads,
            info.num_kv_heads,
            info.hidden_size,
            info.head_dim
        ),
        (Some(36), Some(32), Some(8), Some(4096), Some(128))
    );
    assert!(info.max_position_embeddings == Some(40960) && info.num_params == Some(8_190_000_000));
    assert_eq!(info.weight_bytes, Some(5_000_000_000));
    assert!(
        info.quantization.as_deref() == Some("q4_k_m")
            && !info.is_moe
            && a.ops("models.config").is_empty()
    );
    w.stop().await;
}

#[test]
fn test_gguf_info_reads_llama_cpp_key_names_flat_and_nested() {
    let flat = json!({"general.architecture": "llama", "llama.block_count": 32, "llama.attention.head_count": 32,
                      "llama.attention.head_count_kv": 8, "llama.embedding_length": 4096, "llama.context_length": 131072,
                      "llama.attention.key_length": 128, "llama.expert_count": 8, "general.file_type": 18,
                      "general.size_label": "8B", "tokenizer.ggml.tokens": ["a", "a", "a"], "general.name": "x"});
    let nested = json!({"general": {"architecture": "llama", "file_type": 18, "size_label": "8B"},
                        "llama": {"block_count": 32, "attention": {"head_count": 32, "head_count_kv": 8, "key_length": 128},
                                  "embedding_length": 4096, "context_length": 131072, "expert_count": 8}});
    for meta in [&flat, &nested] {
        let e = obj(json!({"meta": meta, "size": 123}));
        let info = model_info_from_entry("m", &e, "gguf", None);
        assert_eq!(
            (
                info.num_layers,
                info.num_attention_heads,
                info.num_kv_heads,
                info.head_dim
            ),
            (Some(32), Some(32), Some(8), Some(128)),
            "{meta}"
        );
        assert!(
            info.max_position_embeddings == Some(131_072)
                && info.is_moe
                && info.quantization.as_deref() == Some("q6_k")
        );
        assert!(
            info.num_params == Some(8_000_000_000)
                && info.weight_bytes == Some(123)
                && info.architectures == ["llama"]
        );
    }
    let info = model_info_from_entry("m", &obj(json!({"meta": flat})), "gguf", None);
    assert!(!info.raw_config.contains_key("tokenizer.ggml.tokens"));
}

#[test]
fn test_gguf_info_handles_per_layer_arrays_and_missing_keys() {
    let info = model_info_from_entry(
        "m",
        &obj(
            json!({"arch": "hybrid", "meta": {"hybrid.block_count": 4, "hybrid.attention.head_count": [0, 8, 0, 8],
                                               "hybrid.attention.head_count_kv": [0, 2, 0, 4]}}),
        ),
        "gguf",
        None,
    );
    assert!(
        info.num_attention_heads == Some(8)
            && info.num_kv_heads == Some(4)
            && info.head_dim.is_none()
    );
    let bare = model_info_from_entry("m", &obj(json!({})), "gguf", None);
    assert!(
        bare.num_layers.is_none()
            && bare.num_kv_heads.is_none()
            && bare.quantization.is_none()
            && bare.architectures.is_empty()
    );
    assert!(
        model_info_from_entry(
            "m",
            &obj(json!({"meta": "not a dict", "size": -5})),
            "gguf",
            None
        )
        .weight_bytes
        .is_none()
    );
    let mha = model_info_from_entry(
        "m",
        &obj(
            json!({"arch": "a", "meta": {"a.attention.head_count": 16, "a.embedding_length": 2048}}),
        ),
        "gguf",
        None,
    );
    assert!(mha.num_kv_heads == Some(16) && mha.head_dim == Some(128));
}

#[test]
fn test_quant_helpers() {
    assert_eq!(quant_from_file_type(&json!(15)).as_deref(), Some("Q4_K_M"));
    assert_eq!(quant_from_file_type(&json!(32)).as_deref(), Some("BF16"));
    assert!(quant_from_file_type(&json!(999)).is_none());
    assert!(
        quant_from_file_type(&json!(true)).is_none()
            && quant_from_file_type(&json!("15")).is_none()
    );
    assert_eq!(normalize_quant(Some("Q4_K_M")).as_deref(), Some("q4_k_m"));
    assert_eq!(
        normalize_quant(Some("UD-Q4_K_XL")).as_deref(),
        Some("q4_k_xl")
    );
    assert_eq!(normalize_quant(Some("MXFP4_MOE")).as_deref(), Some("mxfp4"));
    assert_eq!(
        normalize_quant(Some("i1-Q4_K_M")).as_deref(),
        Some("q4_k_m")
    );
    assert!(normalize_quant(Some("")).is_none());
    assert_eq!(parse_size_label(&json!("8.0B")), Some(8_000_000_000));
    assert_eq!(parse_size_label(&json!("335M")), Some(335_000_000));
    assert!(parse_size_label(&json!("x")).is_none() && parse_size_label(&Value::Null).is_none());
}

#[test]
fn test_shard_names() {
    assert_eq!(
        shard_names("m/x-00001-of-00003.gguf", None),
        [
            "m/x-00001-of-00003.gguf",
            "m/x-00002-of-00003.gguf",
            "m/x-00003-of-00003.gguf"
        ]
    );
    assert_eq!(
        shard_names("x-00001-of-00002.gguf", Some(2)),
        ["x-00001-of-00002.gguf", "x-00002-of-00002.gguf"]
    );
    assert_eq!(
        shard_names("x-00001-of-00002.gguf", Some(5)),
        ["x-00001-of-00002.gguf"]
    );
    assert_eq!(
        shard_names("x-00002-of-00003.gguf", None),
        ["x-00002-of-00003.gguf"]
    );
    assert_eq!(shard_names("plain.gguf", None), ["plain.gguf"]);
    assert_eq!(
        shard_names("x-00001-of-00001.gguf", None),
        ["x-00001-of-00001.gguf"]
    );
}

#[tokio::test]
async fn test_ollama_info_from_the_config_blob_facts() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path());
    a.st().entries = vec![entry(
        "ol",
        "models/manifests/registry.ollama.ai/library/llama3/8b",
        "ollama",
        Some("llama3:8b"),
        json!({"size": 4_700_000_000_i64,
               "meta": {"model_format": "gguf", "model_family": "llama", "model_type": "8.0B", "file_type": "Q4_0"}}),
    )];
    let w = world(t.path(), vec![a]).await;
    let rf = ModelRef {
        format: "ollama".into(),
        repo_id: None,
        root_id: None,
        rel: None,
        name: Some("llama3:8b".into()),
    };
    let info = w.library.model_info("alpha", &rf).await.unwrap().unwrap();
    assert!(
        info.repo_id == "llama3:8b"
            && info.model_type.as_deref() == Some("llama")
            && info.num_params == Some(8_000_000_000)
    );
    assert!(
        info.quantization.as_deref() == Some("q4_0")
            && info.weight_bytes == Some(4_700_000_000)
            && info.num_layers.is_none()
    );
    w.stop().await;
}

#[tokio::test]
async fn test_no_info_without_a_copy_or_for_other_formats() {
    let t = tempfile::tempdir().unwrap();
    let a = n(t.path());
    a.st().entries = vec![entry("ro", "stuff", "other", Some("stuff"), json!({}))];
    let w = world(t.path(), vec![a]).await;
    assert!(
        w.library
            .model_info("alpha", &ModelRef::hf("org/none"))
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        w.library
            .model_info("alpha", &r("plain", "ro", "stuff"))
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        w.library
            .model_info("nobody", &ModelRef::hf("org/none"))
            .await
            .unwrap()
            .is_none()
    );
    w.stop().await;
}

#[tokio::test]
async fn test_a_node_that_cannot_answer_gives_a_minimal_info_not_an_error() {
    for fmt in ["hf", "plain"] {
        let t = tempfile::tempdir().unwrap();
        let a = n(t.path());
        a.st().entries = if fmt == "hf" {
            vec![hf_entry("hf", "org/m", json!({"size": 9}))]
        } else {
            vec![entry("ro", "d", "plain", Some("d"), json!({"size": 9}))]
        };
        let w = world(t.path(), vec![a.clone()]).await;
        a.st().state = "down".into();
        let rf = if fmt == "hf" {
            ModelRef::hf("org/m")
        } else {
            r("plain", "ro", "d")
        };
        let info = w.library.model_info("alpha", &rf).await.unwrap().unwrap();
        assert!(
            info.weight_bytes == Some(9) && info.num_layers.is_none(),
            "{fmt}"
        );
        w.stop().await;
    }
}
