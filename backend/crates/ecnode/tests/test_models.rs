//! node/tests/test_models.py: the model scanner: every root kind, blob sizes counted once, bounded metadata,
//! budgets, generations.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    unreachable_pub
)]

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ec_proto::Obj;
use ecnode::models::{self, Budget, Scanner, ScannerOptions, model_config, scan_root};
use ecnode::policy::{Root, parse};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const REV: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const STR: u32 = 8;
const U32: u32 = 4;
const ARR: u32 = 9;
const F32: u32 = 6;
const BOOL: u32 = 7;

// ── test_gguf.build ──────────────────────────────────────────────────────────────────────────────────────
fn s(text: &str) -> Vec<u8> {
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend(text.as_bytes());
    out
}

fn kv(key: &str, vtype: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = s(key);
    out.extend(vtype.to_le_bytes());
    out.extend(payload);
    out
}

fn build_with(tensors: Option<Vec<(&str, Vec<u64>)>>, file_type: u32) -> Vec<u8> {
    let arch = "llama";
    let tensors = tensors.unwrap_or_else(|| {
        vec![
            ("tok_embd", vec![4096, 32000]),
            ("blk.0.attn_q", vec![4096, 4096]),
        ]
    });
    let u = |v: u32| v.to_le_bytes().to_vec();
    let mut arr_str = STR.to_le_bytes().to_vec();
    arr_str.extend(3u64.to_le_bytes());
    for t in ["a", "b", "c"] {
        arr_str.extend(s(t));
    }
    let mut arr_f32 = F32.to_le_bytes().to_vec();
    arr_f32.extend(3u64.to_le_bytes());
    for f in [0.0f32, 1.0, 2.0] {
        arr_f32.extend(f.to_le_bytes());
    }
    let kvs = [
        kv("general.architecture", STR, &s(arch)),
        kv("general.name", STR, &s("Test Model 8B")),
        kv("general.file_type", U32, &u(file_type)),
        kv(&format!("{arch}.context_length"), U32, &u(131_072)),
        kv(&format!("{arch}.block_count"), U32, &u(32)),
        kv(&format!("{arch}.embedding_length"), U32, &u(4096)),
        kv(&format!("{arch}.attention.head_count"), U32, &u(32)),
        kv(&format!("{arch}.attention.head_count_kv"), U32, &u(8)),
        kv("tokenizer.ggml.tokens", ARR, &arr_str),
        kv("tokenizer.ggml.scores", ARR, &arr_f32),
        kv("general.flag", BOOL, b"\x01"),
    ];
    let mut head = b"GGUF".to_vec();
    head.extend(3u32.to_le_bytes());
    head.extend((tensors.len() as u64).to_le_bytes());
    head.extend((kvs.len() as u64).to_le_bytes());
    for k in &kvs {
        head.extend(k);
    }
    for (name, dims) in &tensors {
        head.extend(s(name));
        head.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            head.extend(d.to_le_bytes());
        }
        head.extend(12u32.to_le_bytes());
        head.extend(0u64.to_le_bytes());
    }
    head
}

fn build() -> Vec<u8> {
    build_with(None, 15)
}

// ── fixtures ─────────────────────────────────────────────────────────────────────────────────────────────
fn safetensors(shapes: &[(&str, Value)]) -> Vec<u8> {
    let mut hdr = serde_json::Map::new();
    hdr.insert("__metadata__".into(), json!({"format": "pt"}));
    for (k, v) in shapes {
        hdr.insert(
            (*k).to_string(),
            json!({"dtype": "BF16", "shape": v, "data_offsets": [0, 0]}),
        );
    }
    let hdr = Value::Object(hdr).to_string().into_bytes();
    let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
    out.extend(hdr);
    out
}

fn mkdirs(p: &Path) {
    std::fs::create_dir_all(p).unwrap();
}

fn write(p: &Path, data: &[u8]) {
    mkdirs(p.parent().unwrap());
    std::fs::write(p, data).unwrap();
}

/// test_models.hf_repo: blobs, refs/main and a snapshot of links; with extra_snapshot an older revision sharing the
/// blobs (counted once).
fn hf_repo(hub: &Path, repo_id: &str, files: &[(&str, Vec<u8>)], extra_snapshot: bool) -> PathBuf {
    let repo = hub.join(format!("models--{}", repo_id.replace('/', "--")));
    mkdirs(&repo.join("blobs"));
    write(&repo.join("refs").join("main"), REV.as_bytes());
    let snap = repo.join("snapshots").join(REV);
    mkdirs(&snap);
    for (name, content) in files {
        let blob = format!("blob{}", &hex::encode(Sha256::digest(content))[..12]);
        std::fs::write(repo.join("blobs").join(&blob), content).unwrap();
        let target = snap.join(name);
        mkdirs(target.parent().unwrap());
        let depth = name.matches('/').count();
        symlink(format!("{}blobs/{blob}", "../".repeat(2 + depth)), &target).unwrap();
    }
    if extra_snapshot {
        let old = repo.join("snapshots").join("b".repeat(40));
        mkdirs(&old);
        for (name, _) in files {
            let link = std::fs::read_link(snap.join(name)).unwrap();
            symlink(link, old.join(name.rsplit('/').next().unwrap())).unwrap();
        }
    }
    repo
}

fn root_at(path: &Path, kind: &str) -> Root {
    Root {
        id: "r".into(),
        path: path.to_string_lossy().into_owned(),
        kind: kind.into(),
        writable: true,
        label: "r".into(),
    }
}

fn entries_of(r: &Root) -> BTreeMap<String, Obj> {
    let (found, err) = scan_root(r, Budget::default()).unwrap();
    assert_eq!(err, None);
    found
        .into_iter()
        .map(|e| (e["rel"].as_str().unwrap().to_string(), e))
        .collect()
}

fn warned(e: &Obj, needle: &str) -> bool {
    e["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|w| w.as_str().unwrap().contains(needle))
}

fn tmp() -> (tempfile::TempDir, PathBuf) {
    let t = tempfile::tempdir().unwrap();
    let p = std::fs::canonicalize(t.path()).unwrap();
    (t, p)
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_hf_home_repo_sizes_blobs_once_and_reads_bounded_metadata() {
    let (_t, tmp) = tmp();
    let cfg = json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "max_position_embeddings": 131_072,
                     "num_hidden_layers": 32, "torch_dtype": "bfloat16", "unrelated_big_field": "x".repeat(10000),
                     "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128], "noise": 1}});
    let cfg_bytes = cfg.to_string().into_bytes();
    let weights = safetensors(&[("a", json!([4096, 4096])), ("b", json!([10]))]);
    hf_repo(
        &tmp.join("hf").join("hub"),
        "org/model-a",
        &[
            ("config.json", cfg_bytes.clone()),
            ("model.safetensors", weights.clone()),
            ("tokenizer.json", b"{}".to_vec()),
        ],
        true,
    );
    let found = entries_of(&root_at(&tmp.join("hf"), "hf-home"));
    let e = &found["hub/models--org--model-a"];
    assert_eq!(
        (
            &e["format"],
            &e["name"],
            &e["revision"],
            &e["quant"],
            &e["arch"],
            &e["context"]
        ),
        (
            &json!("hf"),
            &json!("org/model-a"),
            &json!(REV),
            &json!("fp8"),
            &json!("LlamaForCausalLM"),
            &json!(131_072)
        )
    );
    assert_eq!(e["size"], json!(cfg_bytes.len() + weights.len() + 2));
    assert_eq!(e["files"], json!(3));
    assert_eq!(e["params"], json!(4096 * 4096 + 10));
    assert_eq!(e["shards"], json!(1));
    assert!(e["meta"].get("unrelated_big_field").is_none());
    assert_eq!(
        e["meta"]["quantization_config"],
        json!({"quant_method": "fp8", "weight_block_size": [128, 128]})
    );
    assert!(warned(e, "2 revisions"));
}

#[test]
fn test_hf_hub_layout_auto_detected_and_quant_from_name_and_downloads_in_progress() {
    let (_t, tmp) = tmp();
    let repo = hf_repo(
        &tmp.join("hub"),
        "org/Model-NVFP4",
        &[
            ("config.json", br#"{"model_type": "x"}"#.to_vec()),
            ("model.safetensors", b"w".repeat(10)),
        ],
        false,
    );
    std::fs::write(repo.join("blobs").join("zzz.incomplete"), b"partial").unwrap();
    let found = entries_of(&root_at(&tmp.join("hub"), "auto"));
    let e = &found["models--org--Model-NVFP4"];
    assert_eq!(e["quant"], json!("nvfp4"));
    assert_eq!(e["params"], Value::Null); // packed 4-bit: no element count
    assert!(warned(e, "still downloading"));
}

#[test]
fn test_gguf_only_hf_repo_is_one_entry_at_the_repo_dir_listing_its_quants() {
    let (_t, tmp) = tmp();
    let hub = tmp.join("hf").join("hub");
    hf_repo(
        &hub,
        "org/Model-GGUF",
        &[
            ("Model-Q4_K_M.gguf", build()),
            ("Model-Q8_0.gguf", build_with(None, 7)),
        ],
        false,
    );
    hf_repo(
        &hub,
        "org/Solo-GGUF",
        &[("Solo-Q5_K_M.gguf", build_with(None, 17))],
        false,
    );
    let found = entries_of(&root_at(&tmp.join("hf"), "hf-home"));
    assert_eq!(
        found.keys().cloned().collect::<BTreeSet<_>>(),
        ["hub/models--org--Model-GGUF", "hub/models--org--Solo-GGUF"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    );
    let e = &found["hub/models--org--Model-GGUF"];
    assert_eq!(
        (
            &e["format"],
            &e["name"],
            &e["revision"],
            &e["quant"],
            &e["arch"]
        ),
        (
            &json!("gguf"),
            &json!("org/Model-GGUF"),
            &json!(REV),
            &Value::Null,
            &json!("llama")
        )
    );
    let files: BTreeMap<String, Value> = e["meta"]["gguf_files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| (f["quant"].as_str().unwrap().to_string(), f.clone()))
        .collect();
    assert_eq!(
        files.keys().cloned().collect::<Vec<_>>(),
        vec!["Q4_K_M", "Q8_0"]
    );
    assert_eq!(e["meta"]["hf_repo"], json!("org/Model-GGUF"));
    assert_eq!(
        files["Q8_0"]["rel"],
        json!(format!("snapshots/{REV}/Model-Q8_0.gguf"))
    );
    assert_eq!(files["Q8_0"]["files"], json!(1));
    let solo = &found["hub/models--org--Solo-GGUF"];
    assert_eq!(solo["quant"], json!("Q5_K_M"));
    assert_eq!(solo["params"], json!(4096 * 32000 + 4096 * 4096));
    let hub_root = entries_of(&root_at(&hub, "hf-hub"));
    assert_eq!(
        hub_root.keys().cloned().collect::<Vec<_>>(),
        vec!["models--org--Model-GGUF", "models--org--Solo-GGUF"]
    ); // hf-hub: no hub/
}

#[test]
fn test_gguf_root_groups_split_sets() {
    let (_t, tmp) = tmp();
    let g = tmp.join("g");
    mkdirs(&g.join("deep").join("er"));
    write(&g.join("Qwen3-8B-Q4_K_M.gguf"), &build());
    write(
        &g.join("deep/er/Big-UD-Q2_K_XL-00001-of-00003.gguf"),
        &build(),
    );
    write(
        &g.join("deep/er/Big-UD-Q2_K_XL-00002-of-00003.gguf"),
        &build_with(Some(vec![("x", vec![10])]), 15),
    );
    write(&g.join(".hidden").join("skip.gguf"), &build());
    let found = entries_of(&root_at(&g, "gguf"));
    let single = &found["Qwen3-8B-Q4_K_M.gguf"];
    assert_eq!(
        (
            &single["name"],
            &single["quant"],
            &single["context"],
            &single["arch"]
        ),
        (
            &json!("Qwen3-8B"),
            &json!("Q4_K_M"),
            &json!(131_072),
            &json!("llama")
        )
    );
    let split = &found["deep/er/Big-UD-Q2_K_XL-00001-of-00003.gguf"];
    assert_eq!((&split["shards"], &split["files"]), (&json!(3), &json!(2)));
    assert!(warned(split, "2 of 3"));
    assert_eq!(split["params"], json!(4096 * 32000 + 4096 * 4096 + 10));
    assert_eq!(
        split["meta"]["files"],
        json!([
            "Big-UD-Q2_K_XL-00001-of-00003.gguf",
            "Big-UD-Q2_K_XL-00002-of-00003.gguf"
        ])
    );
    assert!(!found.keys().any(|k| k.contains(".hidden")));
}

#[test]
fn test_plain_root_and_secret_files_are_skipped() {
    let (_t, tmp) = tmp();
    let p = tmp.join("p");
    let m = p.join("team").join("llama-ft");
    write(
        &m.join("config.json"),
        br#"{"architectures": ["X"], "max_position_embeddings": 4096}"#,
    );
    write(
        &m.join("model.safetensors"),
        &safetensors(&[("w", json!([2, 3]))]),
    );
    write(&m.join("credentials.json"), b"never read");
    write(&p.join("notes").join("config.json"), b"{}"); // config without weights: not a model
    let found = entries_of(&root_at(&p, "plain"));
    let e = &found["team/llama-ft"];
    assert_eq!(
        (&e["format"], &e["name"], &e["params"], &e["context"]),
        (&json!("plain"), &json!("llama-ft"), &json!(6), &json!(4096))
    );
    assert_eq!(e["files"], json!(2)); // credentials.json is not even counted
}

#[test]
fn test_plain_dir_with_weights_in_an_unknown_format_is_listed_with_a_warning() {
    let (_t, tmp) = tmp();
    let p = tmp.join("p");
    let m = p.join("Custom-Model");
    write(&m.join("config.json"), br#"{"architectures": ["X"]}"#);
    let f = std::fs::File::create(m.join("custom-00000.ftw")).unwrap(); // sparse: large on paper
    f.set_len(models::BIG_FILE + 1).unwrap();
    write(&m.join("small-00001.ftw"), &b"x".repeat(10)); // small unknown files alone would not make it a model
    let found = entries_of(&root_at(&p, "plain"));
    let e = &found["Custom-Model"];
    assert_eq!(e["format"], json!("plain"));
    assert!(e["size"].as_u64().unwrap() >= models::BIG_FILE);
    assert!(e["warnings"].as_array().unwrap().iter().any(
        |w| w.as_str().unwrap().contains(".ftw") && w.as_str().unwrap().contains("do not load")
    ));
    let tiny = p.join("Tiny");
    write(&tiny.join("config.json"), b"{}");
    write(&tiny.join("x.ftw"), b"x");
    assert!(!entries_of(&root_at(&p, "plain")).contains_key("Tiny"));
}

/// Python records every os.open; Rust cannot patch the syscall, so the key and history files are FIFOs here: any
/// blocking open would hang the scan (a non-blocking open only happens through open_inside, which refuses
/// secret names and anything that is not a regular file), and neither name may show up in any entry.
#[test]
fn test_ollama_home_never_touches_its_keys() {
    let (_t, tmp) = tmp();
    let home = tmp.join(".ollama");
    let blobs = home.join("models").join("blobs");
    mkdirs(&blobs);
    for trap in ["id_ed25519", "history"] {
        nix::unistd::mkfifo(
            &home.join(trap),
            nix::sys::stat::Mode::from_bits_truncate(0o600),
        )
        .unwrap();
    }
    let model = build();
    let cfg = json!({"model_format": "gguf", "model_family": "llama", "model_type": "8.0B", "file_type": "Q4_K_M"})
        .to_string()
        .into_bytes();
    let (dm, dc) = ("1".repeat(64), "2".repeat(64));
    std::fs::write(blobs.join(format!("sha256-{dm}")), &model).unwrap();
    std::fs::write(blobs.join(format!("sha256-{dc}")), &cfg).unwrap();
    let man = home.join("models/manifests/registry.ollama.ai/library/llama3");
    mkdirs(&man);
    let manifest = json!({"config": {"digest": format!("sha256:{dc}"), "size": cfg.len()},
                          "layers": [{"mediaType": "application/vnd.ollama.image.model",
                                      "digest": format!("sha256:{dm}"), "size": model.len()},
                                     {"mediaType": "application/vnd.ollama.image.license",
                                      "digest": format!("sha256:{}", "3".repeat(64)), "size": 10}]});
    std::fs::write(man.join("8b"), manifest.to_string()).unwrap();
    let hf = home.join("models/manifests/hf.co/org/repo");
    mkdirs(&hf);
    std::fs::write(hf.join("Q8_0"), r#"{"layers": []}"#).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let (h1, h2) = (home.clone(), home.join("models"));
    std::thread::spawn(move || {
        let found = entries_of(&root_at(&h1, "ollama"));
        let by_models_dir = entries_of(&root_at(&h2, "auto")); // a root at the models/ dir itself
        let _ = tx.send((found, by_models_dir));
    });
    let (found, by_models_dir) = rx
        .recv_timeout(Duration::from_secs(30))
        .expect("the scan blocked: it opened a key or history file");
    assert_eq!(
        by_models_dir.keys().cloned().collect::<BTreeSet<_>>(),
        [
            "manifests/hf.co/org/repo/Q8_0",
            "manifests/registry.ollama.ai/library/llama3/8b"
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    );
    let e = &found["models/manifests/registry.ollama.ai/library/llama3/8b"];
    assert_eq!(
        (
            &e["format"],
            &e["name"],
            &e["quant"],
            &e["arch"],
            &e["context"]
        ),
        (
            &json!("ollama"),
            &json!("llama3:8b"),
            &json!("Q4_K_M"),
            &json!("llama"),
            &json!(131_072)
        )
    );
    assert_eq!(e["size"], json!(model.len() + 10));
    assert!(warned(e, "missing"));
    assert_eq!(
        found["models/manifests/hf.co/org/repo/Q8_0"]["name"],
        json!("hf.co/org/repo:Q8_0")
    );
    let blob = serde_json::to_string(&found.values().collect::<Vec<_>>()).unwrap();
    assert!(!blob.contains("id_ed25519") && !blob.contains("history"));
}

#[test]
fn test_other_root_and_auto_fallback() {
    let (_t, tmp) = tmp();
    let o = tmp.join("o");
    write(&o.join("a").join("b").join("f"), &b"x".repeat(100));
    write(&o.join("a").join("token"), b"secret");
    let found = entries_of(&root_at(&o, "other"));
    let e = &found["a"];
    assert_eq!(
        (&e["format"], &e["size"], &e["files"]),
        (&json!("other"), &json!(100), &json!(1))
    );
    assert!(entries_of(&root_at(&o, "auto")).is_empty()); // no models found by plain + gguf
}

#[test]
fn test_budget_exhaustion_keeps_what_was_found() {
    let (_t, tmp) = tmp();
    let g = tmp.join("g");
    mkdirs(&g.join("a"));
    mkdirs(&g.join("b"));
    for i in 0..3 {
        write(&g.join("a").join(format!("m{i}-Q4_0.gguf")), &build());
    }
    for i in 0..30 {
        write(&g.join("b").join(format!("m{i:02}-Q4_0.gguf")), &build());
    }
    let (found, err) = scan_root(&root_at(&g, "gguf"), Budget::new(10, 120.0)).unwrap();
    assert!(
        err.as_deref().unwrap().contains("10 directory entries"),
        "{err:?}"
    );
    let mut rels: Vec<String> = found
        .iter()
        .map(|e| e["rel"].as_str().unwrap().to_string())
        .collect();
    rels.sort();
    assert_eq!(
        rels,
        (0..3)
            .map(|i| format!("a/m{i}-Q4_0.gguf"))
            .collect::<Vec<_>>()
    );
    let (_found2, err2) = scan_root(&root_at(&g, "gguf"), Budget::new(1000, 0.0)).unwrap();
    assert!(
        err2.as_deref().unwrap().contains("s (scan budget)"),
        "{err2:?}"
    );
}

#[test]
fn test_symlinked_files_leaving_the_root_are_ignored() {
    let (_t, tmp) = tmp();
    let out = tmp.join("outside.gguf");
    std::fs::write(&out, build()).unwrap();
    let g = tmp.join("g");
    mkdirs(&g);
    symlink(&out, g.join("escape.gguf")).unwrap();
    assert!(entries_of(&root_at(&g, "gguf")).is_empty());
}

// ── models.config ────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_model_config_for_hf_plain_gguf_and_refusals() {
    let (_t, tmp) = tmp();
    hf_repo(
        &tmp.join("hf").join("hub"),
        "org/m",
        &[(
            "config.json",
            br#"{"model_type": "llama", "n": 1}"#.to_vec(),
        )],
        false,
    );
    write(
        &tmp.join("pl").join("m").join("config.json"),
        br#"{"a": 1}"#,
    );
    write(&tmp.join("pl").join("f.gguf"), &build());
    write(&tmp.join("secret.json"), br#"{"leak": true}"#);
    mkdirs(&tmp.join("pl").join("esc"));
    symlink(
        tmp.join("secret.json"),
        tmp.join("pl").join("esc").join("config.json"),
    )
    .unwrap();
    let p = parse(
        &json!({"model_roots": [{"id": "hf", "path": tmp.join("hf"), "kind": "hf-home"},
                                {"id": "pl", "path": tmp.join("pl"), "kind": "plain"}]}),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        model_config(&p, &json!("hf"), Some("hub/models--org--m")).unwrap()["config"],
        json!({"model_type": "llama", "n": 1})
    );
    assert_eq!(
        model_config(&p, &json!("pl"), Some("m")).unwrap()["config"],
        json!({"a": 1})
    );
    assert_eq!(
        model_config(&p, &json!("pl"), Some("f.gguf")).unwrap()["config"]["gguf"]["arch"],
        json!("llama")
    );
    for rel in ["esc", "nope"] {
        match model_config(&p, &json!("pl"), Some(rel)) {
            Err(ecnode::errors::HandlerError::Op(e)) => {
                assert_eq!(e.code, ec_proto::NOT_FOUND, "{rel}");
            }
            other => panic!("{rel}: {other:?}"),
        }
    }
    assert!(matches!(
        model_config(&p, &json!("pl"), Some("../hf")),
        Err(ecnode::errors::HandlerError::Op(_))
    ));
}

// ── the background scanner ───────────────────────────────────────────────────────────────────────────────
fn fixed_policy(p: ecnode::policy::Policy) -> models::PolicyFn {
    let p = Arc::new(p);
    Arc::new(move || Arc::clone(&p))
}

fn recorder() -> (models::OnChange, Arc<Mutex<Vec<i64>>>) {
    let seen: Arc<Mutex<Vec<i64>>> = Arc::default();
    let s = Arc::clone(&seen);
    (Arc::new(move |g| s.lock().unwrap().push(g)), seen)
}

#[test]
fn test_scanner_generations_paging_and_publishing() {
    let (_t, tmp) = tmp();
    let g = tmp.join("g");
    mkdirs(&g);
    for i in 0..5 {
        write(&g.join(format!("m{i}-Q4_0.gguf")), &build());
    }
    let pol = parse(
        &json!({"model_roots": [{"id": "g", "path": g, "kind": "gguf"},
                                {"id": "missing", "path": tmp.join("nope"), "kind": "gguf"}]}),
        None,
        None,
    )
    .unwrap();
    let (on_change, changed) = recorder();
    let sc = Scanner::new(fixed_policy(pol), on_change, ScannerOptions::default());
    assert_eq!(sc.page(None, None)["generation"], json!(0));
    let generation = sc.request(None, true);
    assert_eq!(sc.run_once(), Some(generation));
    assert_eq!(*changed.lock().unwrap(), vec![generation]);
    let p1 = sc.page(Some(0), Some(2));
    assert_eq!(p1["generation"], json!(generation));
    assert_eq!(p1["entries"].as_array().unwrap().len(), 2);
    assert_eq!(p1["next_cursor"], json!(2));
    let p3 = sc.page(Some(4), Some(2));
    assert_eq!(p3["entries"].as_array().unwrap().len(), 1);
    assert_eq!(p3["next_cursor"], Value::Null);
    let scans: BTreeMap<String, Value> = p1["roots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| (r["root"].as_str().unwrap().to_string(), r.clone()))
        .collect();
    assert_eq!(
        (&scans["g"]["complete"], &scans["g"]["entries"]),
        (&json!(true), &json!(5))
    );
    assert!(
        scans["missing"]["error"]
            .as_str()
            .unwrap()
            .contains("does not exist")
    );
    // an automatic rescan that finds nothing new publishes nothing ...
    sc.queue_full_rescan();
    assert_eq!(sc.run_once(), None);
    assert_eq!(*changed.lock().unwrap(), vec![generation]);
    // ... a requested one always publishes the generation it promised
    let promised = sc.request(None, true);
    assert!(promised > generation);
    assert_eq!(sc.run_once(), Some(promised));
    assert_eq!(changed.lock().unwrap().last(), Some(&promised));
    // a change is published
    write(&g.join("m9-Q8_0.gguf"), &build_with(None, 7));
    sc.queue_full_rescan();
    let newer = sc.run_once().unwrap();
    let page = sc.page(Some(0), Some(200));
    assert_eq!(page["generation"], json!(newer));
    assert_eq!(page["entries"].as_array().unwrap().len(), 6);
}

#[test]
fn test_scanner_answers_a_fresh_request_with_the_current_generation() {
    let pol = parse(&json!({"model_roots": []}), None, None).unwrap();
    let sc = Scanner::new(
        fixed_policy(pol),
        Arc::new(|_| {}),
        ScannerOptions::default(),
    );
    sc.request(None, true);
    let generation = sc.run_once().unwrap();
    assert_eq!(sc.request(None, false), generation); // fresh: no new scan
    assert_eq!(sc.request(None, true), generation + 1);
}

#[test]
fn test_hung_root_is_reported_and_the_previous_result_kept() {
    let (_t, tmp) = tmp();
    let g = tmp.join("g");
    write(&g.join("a-Q4_0.gguf"), &build());
    let pol = parse(
        &json!({"model_roots": [{"id": "g", "path": g, "kind": "gguf"}]}),
        None,
        None,
    )
    .unwrap();
    // scan_root, patched: once `block` is set, each call first waits (at most 5 s) for the gate
    let block = Arc::new(AtomicBool::new(false));
    let gate: Arc<(Mutex<bool>, Condvar)> = Arc::default();
    let hook: models::ScanRootFn = {
        let (block, gate) = (Arc::clone(&block), Arc::clone(&gate));
        Arc::new(move |r: &Root, b: Budget| {
            if block.load(Ordering::SeqCst) {
                let (m, cv) = &*gate;
                let open = m.lock().unwrap();
                let _ = cv
                    .wait_timeout_while(open, Duration::from_secs(5), |o| !*o)
                    .unwrap();
            }
            scan_root(r, b)
        })
    };
    let opts = ScannerOptions {
        seconds: 0.1,
        hung_grace_s: 0.2,
        scan_root: Some(hook),
        ..ScannerOptions::default()
    };
    let sc = Scanner::new(fixed_policy(pol), Arc::new(|_| {}), opts);
    sc.request(None, true);
    sc.run_once();
    block.store(true, Ordering::SeqCst);
    sc.request(None, true);
    let t0 = Instant::now();
    sc.run_once();
    {
        let (m, cv) = &*gate;
        *m.lock().unwrap() = true;
        cv.notify_all();
    }
    assert!(t0.elapsed() < Duration::from_secs(3));
    let page = sc.page(Some(0), Some(10));
    assert_eq!(page["entries"].as_array().unwrap().len(), 1);
    assert!(
        page["roots"][0]["error"]
            .as_str()
            .unwrap()
            .contains("did not answer")
    );
}

#[test]
fn test_m3_a_hostile_safetensors_header_is_refused_without_burning_cpu() {
    let (_t, tmp) = tmp();
    let one = |shape: Value| -> Option<i64> {
        let f = tmp.join("w.safetensors");
        std::fs::write(&f, safetensors(&[("a", shape)])).unwrap();
        models::safetensors_params(&tmp, &[f], None)
    };
    assert_eq!(one(json!([4096, 4096])), Some(4096 * 4096));
    assert_eq!(one(json!(vec![1; 100_000])), None); // far too many dims: refused, whatever they multiply to
    assert_eq!(one(json!([1u64 << 40, 1u64 << 40])), None); // past 2**53 parameters
    let t0 = Instant::now();
    assert_eq!(one(json!(vec![3; 2_000_000])), None); // unbounded, this one is minutes of CPU
    assert!(t0.elapsed() < Duration::from_secs(5));
}
