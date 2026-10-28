//! Port of backend/tests/data/test_library_gguf_repo.py: GGUF-only HF repos (one entry per repo from the node, one
//! model per quant in the library, the quant's header read through the node, the launch mounting the whole repo
//! directory: the lifecycle's runspec `model_mounts`).

#![allow(clippy::unwrap_used)]

#[path = "common/wsc_data.rs"]
mod wsc_data;

use ec_hub::adapters::base::MODEL_MOUNT;
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::libport::LibraryPort as _;
use ec_hub::services::library::expand_gguf_repo;
use ec_hub::services::runspec::model_mounts;
use serde_json::{Value, json};
use wsc_data::*;

const REPO_REL: &str = "models--unsloth--Big-GGUF";

fn q4() -> String {
    format!("snapshots/{COMMIT}/Q4_K_M/Big-Q4_K_M-00001-of-00002.gguf")
}

fn q8() -> String {
    format!("snapshots/{COMMIT}/Big-Q8_0.gguf")
}

fn repo_entry(extra: Value) -> Value {
    let files = json!([
        {"rel": q4(), "quant": "Q4_K_M", "size": 300, "files": 2, "shards": 2, "role": null},
        {"rel": q8(), "quant": "Q8_0", "size": 500, "files": 1, "shards": null, "role": null},
        {"rel": format!("snapshots/{COMMIT}/mmproj-F16.gguf"), "quant": "F16", "size": 9, "files": 1, "shards": null,
         "role": "mmproj"}]);
    let mut e = json!({"root": "gguf", "rel": REPO_REL, "format": "gguf", "name": "unsloth/Big-GGUF", "revision": COMMIT,
                       "size": 809, "files": 4, "mtime": 1.0, "atime": 2.0, "quant": null, "params": null, "arch": "qwen3",
                       "context": 32768, "shards": null, "meta": {"hf_repo": "unsloth/Big-GGUF", "gguf_files": files},
                       "warnings": []});
    if let (Value::Object(a), Value::Object(b)) = (&mut e, extra) {
        for (k, v) in b {
            a.insert(k, v);
        }
    }
    e
}

#[test]
fn test_expand_lists_one_entry_per_quant_and_skips_the_projector() {
    let out = expand_gguf_repo(&obj(repo_entry(json!({}))));
    assert_eq!(
        out.iter()
            .map(|e| e["rel"].as_str().unwrap().to_string())
            .collect::<Vec<_>>(),
        [
            format!("{REPO_REL}/{}", q4()),
            format!("{REPO_REL}/{}", q8())
        ]
    );
    let triples: Vec<Value> = out
        .iter()
        .map(|e| json!([e["quant"], e["size"], e["shards"]]))
        .collect();
    assert_eq!(
        triples,
        [json!(["Q4_K_M", 300, 2]), json!(["Q8_0", 500, null])]
    );
    assert!(
        out.iter()
            .all(|e| e["meta"] == json!({"hf_repo": "unsloth/Big-GGUF", "repo_rel": REPO_REL}))
    );
    assert!(out.iter().all(|e| {
        e["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("mmproj"))
    }));
}

#[test]
fn test_expand_leaves_everything_else_alone() {
    let loose = obj(
        json!({"root": "gguf", "rel": "dir/m-Q4_K_M.gguf", "format": "gguf", "meta": {"block_count": 2}}),
    );
    let hf = obj(
        json!({"root": "hf", "rel": "hub/models--a--b", "format": "hf", "meta": {"gguf_files": [{"rel": "x.gguf"}]}}),
    );
    assert_eq!(expand_gguf_repo(&loose), std::slice::from_ref(&loose));
    assert_eq!(expand_gguf_repo(&hf), std::slice::from_ref(&hf));
    assert_eq!(
        expand_gguf_repo(&obj(repo_entry(json!({"meta": {"gguf_files": []}}))))[0]["rel"],
        REPO_REL
    );
}

#[tokio::test]
async fn test_library_lists_each_quant_and_reads_its_header_through_the_node() {
    let t = tempfile::tempdir().unwrap();
    let a = node(
        t.path(),
        "alpha",
        NodeOpts {
            scan_disk: false,
            roots: vec![RootCfg::new("gguf", "gguf")],
            ..NodeOpts::default()
        },
    );
    a.st().entries = vec![repo_entry(json!({}))];
    a.st().configs.insert(
        ("gguf".into(), format!("{REPO_REL}/{}", q4())),
        json!({"gguf": {"arch": "qwen3", "context": 32768, "params": 30_000_000_000_i64, "block_count": 48,
                        "embedding_length": 4096, "head_count": 32, "head_count_kv": 4}}),
    );
    let w = World::new(t.path(), vec![a.clone()], WorldOpts::default())
        .start()
        .await;
    let mut keys: Vec<String> = w
        .library
        .models(None, Some("gguf"), None)
        .unwrap()
        .into_iter()
        .map(|m| m.key)
        .collect();
    keys.sort();
    assert_eq!(
        keys,
        ["gguf:unsloth/Big-GGUF:Q4_K_M", "gguf:unsloth/Big-GGUF:Q8_0"]
    );
    let rf = ModelRef {
        format: "gguf".into(),
        repo_id: None,
        root_id: Some("gguf".into()),
        rel: Some(format!("{REPO_REL}/{}", q4())),
        name: None,
    };
    let found = w.library.find_copy("alpha", &rf, false).unwrap();
    assert!(found.size_bytes == 300 && found.rel == format!("{REPO_REL}/{}", q4()));
    let info = w.library.model_info("alpha", &rf).await.unwrap().unwrap();
    assert_eq!(
        (info.num_layers, info.num_kv_heads, info.num_params),
        (Some(48), Some(4), Some(30_000_000_000))
    );
    assert!(info.weight_bytes == Some(300) && info.quantization.as_deref() == Some("q4_k_m"));
    assert_eq!(
        a.ops("models.config")
            .into_iter()
            .map(Value::Object)
            .collect::<Vec<_>>(),
        [json!({"root": "gguf", "rel": format!("{REPO_REL}/{}", q4())})]
    );
    w.stop().await;
}

fn gguf(root: &str, rel: &str) -> ModelRef {
    ModelRef {
        format: "gguf".into(),
        repo_id: None,
        root_id: Some(root.into()),
        rel: Some(rel.into()),
        name: None,
    }
}

/// A quant inside an HF cache repo mounts the whole repo directory (its snapshot file links into the repo's blobs).
#[test]
fn test_a_quant_inside_an_hf_repo_mounts_the_repo_directory() {
    let (mounts, path) = model_mounts(&gguf("gguf", &format!("{REPO_REL}/{}", q4())), None, None);
    assert_eq!(
        mounts.into_iter().map(Value::Object).collect::<Vec<_>>(),
        [json!({"root": "gguf", "rel": REPO_REL, "dest": MODEL_MOUNT, "ro": true})]
    );
    assert_eq!(path, Some(format!("{MODEL_MOUNT}/{}", q4())));
    let home = format!("hub/{REPO_REL}/{}", q8());
    let (mounts, path) = model_mounts(&gguf("hf", &home), None, None);
    assert_eq!(mounts[0]["rel"], json!(format!("hub/{REPO_REL}")));
    assert_eq!(path, Some(format!("{MODEL_MOUNT}/{}", q8())));
}

#[test]
fn test_a_loose_gguf_file_still_mounts_its_own_directory() {
    let (mounts, path) = model_mounts(&gguf("gguf", "dir/m-Q4_K_M.gguf"), None, None);
    assert_eq!(
        mounts.into_iter().map(Value::Object).collect::<Vec<_>>(),
        [json!({"root": "gguf", "rel": "dir", "dest": MODEL_MOUNT, "ro": true})]
    );
    assert_eq!(path, Some(format!("{MODEL_MOUNT}/m-Q4_K_M.gguf")));
}
