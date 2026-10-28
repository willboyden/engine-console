//! Port of backend/tests/data/test_live_node.py: the library and the download service on top of two REAL nodes, the
//! Rust `ecnode --stdio` (target/debug/ecnode) reached through the hub's own `NodeLink`.
//!
//! Python is opt-in (ENGINE_CONSOLE_LIVE_NODE=1) and runs its Python node. Here the test runs whenever the ecnode binary
//! has been built (`cargo test --workspace` builds it; else it says SKIP), because it is hermetic: temp roots only, a
//! PATH and an `ECNODE_TOOL_PATH` with no docker and no nvidia-smi, `ECNODE_NO_NVML=1`, policies with `"discovery_ports": []` and a docker binary
//! that does not exist, written 0644 and checked to load (`policy.error` null, PORTING.md §A6), and Hugging Face is the
//! in-memory fake hub. It proves that the real `fs.stat` / `fs.write` / `fs.read` / `fs.symlink` / `fs.put_small` /
//! `fs.delete` / `models.*` behave as the data side assumes (partials, resume offsets, hash checks, relative links, the
//! scanner's entry shapes, GGUF headers).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

#[path = "common/wsc_data.rs"]
mod wsc_data;

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ec_hub::domain::errors::{Problem, not_found};
use ec_hub::domain::fleet::{CopyRequest, FleetPort, HostConfig, NodeClient};
use ec_hub::domain::models::DownloadRequest;
use ec_hub::domain::ports::HubClient;
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::common::EventBus;
use ec_hub::services::downloads::DownloadService;
use ec_hub::services::fleet::{LinkLike, LinkTiming, NodeLink};
use ec_hub::services::hf::HfService;
use ec_hub::services::libport::LibraryPort as _;
use ec_hub::services::library::LibraryService;
use ec_hub::services::settings::SettingsService;
use ec_hub::services::store::Store;
use serde_json::{Value, json};
use sha2::Digest as _;
use wsc_data::{COMMIT, DataHub, Gate, make_adapters, make_settings};

/// target/debug/ecnode, next to this test's target/debug/deps directory.
fn ecnode() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let p = exe.parent()?.parent()?.join("ecnode");
    p.is_file().then_some(p)
}

/// Deterministic noise (Python uses os.urandom): xorshift64.
fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 24) & 0xff).to_le_bytes()[0]
        })
        .collect()
}

fn sha_hex(b: &[u8]) -> String {
    hex::encode(sha2::Sha256::digest(b))
}

/// A minimal valid GGUF v3 file (header + two tensor descriptors, no tensor data): enough for the node's reader.
fn tiny_gguf(path: &Path) {
    fn string(text: &str) -> Vec<u8> {
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(text.as_bytes());
        out
    }
    let u32le = |v: u32| v.to_le_bytes().to_vec();
    let kvs: Vec<(&str, u32, Vec<u8>)> = vec![
        ("general.architecture", 8, string("llama")),
        ("general.name", 8, string("Tiny")),
        ("general.file_type", 4, u32le(15)),
        ("llama.context_length", 4, u32le(4096)),
        ("llama.block_count", 4, u32le(2)),
        ("llama.embedding_length", 4, u32le(64)),
        ("llama.attention.head_count", 4, u32le(4)),
        ("llama.attention.head_count_kv", 4, u32le(2)),
    ];
    let mut out = b"GGUF".to_vec();
    out.extend(u32le(3));
    out.extend(2u64.to_le_bytes());
    out.extend((kvs.len() as u64).to_le_bytes());
    for (k, t, v) in &kvs {
        out.extend(string(k));
        out.extend(u32le(*t));
        out.extend(v);
    }
    for name in ["a.weight", "b.weight"] {
        out.extend(string(name));
        out.extend(u32le(2));
        out.extend(64u64.to_le_bytes());
        out.extend(16u64.to_le_bytes());
        out.extend(u32le(0));
        out.extend(0u64.to_le_bytes());
    }
    std::fs::write(path, out).unwrap();
}

/// test_live_node.Fleet
struct Links(Vec<Arc<NodeLink>>);

impl FleetPort for Links {
    fn hosts(&self) -> Vec<HostConfig> {
        self.0.iter().map(|l| l.host().clone()).collect()
    }
    fn node(&self, host_id: &str) -> Result<Arc<dyn NodeClient>, Problem> {
        self.0
            .iter()
            .find(|l| l.host().id == host_id)
            .map(|l| Arc::clone(l) as Arc<dyn NodeClient>)
            .ok_or_else(|| not_found(format!("no host {host_id}")).code("unknown_host"))
    }
}

/// The hermetic policy (PORTING.md §A6): no discovery probes, no docker, an explicit mode.
fn write_policy(path: &Path, roots: &[Value]) {
    std::fs::write(
        path,
        json!({"version": 1, "mode": "manage", "docker": {"binary": "/nonexistent/docker", "context": null},
               "engine_images": ["vllm/vllm-openai:*"], "model_roots": roots, "scripts": [], "allow_pull": false,
               "allow_delete": true, "allow_ipc_host": false, "external_control": "none", "max_shm_gib": 8,
               "discovery_ports": []})
        .to_string(),
    )
    .unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o644)).unwrap();
}

fn start_link(
    bin: &Path,
    tmp: &Path,
    bus: &Arc<EventBus>,
    host_id: &str,
    label: &str,
    policy: &Path,
) -> Arc<NodeLink> {
    let home = policy.parent().unwrap().to_path_buf();
    let empty = tmp.join("empty-bin");
    std::fs::create_dir_all(&empty).unwrap();
    let env = vec![
        ("HOME".to_string(), home.display().to_string()),
        ("PATH".to_string(), empty.display().to_string()),
        ("LANG".to_string(), "C.UTF-8".to_string()),
        ("ECNODE_NO_NVML".to_string(), "1".to_string()),
        // the node runs tools only from its fixed PATH or this override (PORTING.md §A11): the empty dir, so no docker
        // and no nvidia-smi of this machine is ever run
        ("ECNODE_TOOL_PATH".to_string(), empty.display().to_string()),
    ];
    let host = HostConfig {
        id: host_id.into(),
        label: label.into(),
        ssh: None,
        port: None,
        ssh_config: false,
        enabled: true,
    };
    let argv = vec![
        bin.display().to_string(),
        "--stdio".into(),
        "--policy".into(),
        policy.display().to_string(),
    ];
    Arc::new(NodeLink::new(
        host,
        NodeLink::fixed_argv(argv),
        Some(env),
        Arc::clone(bus),
        LinkTiming::default(),
        None,
        Vec::new(),
    ))
}

async fn wait_for(what: &str, mut cond: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(20);
    while !cond() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn root(id: &str, path: &Path, kind: &str, writable: bool) -> Value {
    json!({"id": id, "path": path.display().to_string(), "kind": kind, "writable": writable, "label": id})
}

fn dreq(repo: &str, host: &str, root_id: &str, allow: Option<&[&str]>) -> DownloadRequest {
    DownloadRequest {
        repo_id: repo.into(),
        revision: None,
        allow_patterns: allow.map(|a| a.iter().map(|s| (*s).to_string()).collect()),
        host_id: host.into(),
        root_id: Some(root_id.into()),
    }
}

fn mref(format: &str, root_id: &str, rel: &str) -> ModelRef {
    ModelRef {
        format: format.into(),
        repo_id: None,
        root_id: Some(root_id.into()),
        rel: Some(rel.into()),
        name: None,
    }
}

fn incompletes(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.to_string_lossy().ends_with(".incomplete") {
                out.push(p);
            }
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[allow(clippy::too_many_lines)]
async fn test_real_nodes_download_copy_resume_verify_delete_and_cancel() {
    let Some(bin) = ecnode() else {
        eprintln!(
            "SKIP: target/debug/ecnode is not built (cargo build -p ecnode, or cargo test --workspace)"
        );
        return;
    };
    let t = tempfile::tempdir().unwrap();
    let tmp = t.path();
    std::fs::set_permissions(tmp, std::fs::Permissions::from_mode(0o700)).unwrap();
    let (a, b) = (tmp.join("a"), tmp.join("b"));
    let roots: Vec<(&str, &str, Vec<Value>)> = vec![
        (
            "alpha",
            "Alpha",
            vec![
                root("hf", &a.join("hf"), "hf-home", true),
                root("gguf", &a.join("gguf"), "gguf", true),
                root("pl", &a.join("pl"), "plain", true),
            ],
        ),
        (
            "beta",
            "Beta",
            vec![
                root("hub", &b.join("hub"), "hf-hub", true),
                root("gguf", &b.join("gguf"), "gguf", true),
                root("ro", &b.join("ro"), "hf-home", false),
                root("pl", &b.join("pl"), "plain", true),
            ],
        ),
    ];
    let bus = Arc::new(EventBus::new());
    let mut links = Vec::new();
    for (host, label, rs) in &roots {
        for r in rs {
            std::fs::create_dir_all(r["path"].as_str().unwrap()).unwrap();
        }
        let dir = tmp.join(host);
        std::fs::create_dir_all(&dir).unwrap();
        let policy = dir.join("policy.json");
        write_policy(&policy, rs);
        links.push(start_link(&bin, tmp, &bus, host, label, &policy));
    }
    for l in &links {
        l.start().await;
    }
    let l2 = links.clone();
    wait_for("both nodes to come up", || {
        l2.iter().all(|l| l.link().state == "up")
    })
    .await;
    for l in &links {
        // the policy loaded: a refused file falls back to the default policy, which probes ports
        let hello = l.latest("hello").unwrap();
        assert!(
            hello["policy"].get("error").is_none_or(Value::is_null),
            "{}",
            hello["policy"]
        );
    }
    let fleet: Arc<dyn FleetPort> = Arc::new(Links(links.clone()));
    let store = Arc::new(Store::memory().unwrap());
    let lib = Arc::new(LibraryService::with_options(
        Arc::clone(&store),
        Arc::clone(&fleet),
        make_adapters(),
        Arc::clone(&bus),
        Arc::new(|_: &str, _: &str| Vec::new()),
        200,
        &[0.2],
        30.0,
    ));
    let hub = DataHub::new();
    let settings = Arc::new(SettingsService::new(
        Arc::clone(&store),
        Arc::new(make_settings(tmp)),
        Box::new(Vec::new),
    ));
    let dl = DownloadService::new(
        Arc::clone(&store),
        Arc::new(hub.clone()) as Arc<dyn HubClient>,
        Arc::new(HfService::with_ttl(
            Arc::new(hub.clone()) as Arc<dyn HubClient>,
            Duration::ZERO,
        )),
        settings,
        Arc::clone(&bus),
        Arc::clone(&fleet),
        Arc::clone(&lib),
        2,
        50,
    );
    lib.start().await;
    lib.wait_idle(Duration::from_secs(20)).await;

    // a download through the real node, spanning several DATA frames
    let payload = noise(3 * 1024 * 1024 + 123, 1);
    let small: &[u8] = br#"{"model_type": "x", "num_hidden_layers": 2}"#;
    hub.add(
        "org/m",
        &[
            ("config.json", small),
            ("model.safetensors", &payload),
            ("tok.txt", b"tok"),
        ],
    );
    let d = dl
        .create(&dreq("org/m", "alpha", "hf", None))
        .await
        .unwrap();
    dl.wait(&d.id).await;
    let d = dl.get(&d.id).unwrap();
    assert_eq!(d.state, "completed", "{:?} {:?}", d.error_code, d.error);
    let repo = a.join("hf").join("hub").join("models--org--m");
    let sha = sha_hex(&payload);
    assert_eq!(
        std::fs::read(repo.join("blobs").join(&sha)).unwrap(),
        payload
    );
    assert_eq!(
        std::fs::read_link(
            repo.join("snapshots")
                .join(COMMIT)
                .join("model.safetensors")
        )
        .unwrap(),
        PathBuf::from(format!("../../blobs/{sha}"))
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("refs").join("main")).unwrap(),
        COMMIT
    );
    let hf = ModelRef::hf("org/m");
    let l3 = Arc::clone(&lib);
    wait_for("the real scanner to list the download", || {
        l3.find_copy("alpha", &ModelRef::hf("org/m"), false)
            .is_some()
    })
    .await;
    let copy = lib.find_copy("alpha", &hf, false).unwrap();
    assert_eq!(copy.rel, "hub/models--org--m");
    assert_eq!(copy.size_bytes, (payload.len() + small.len() + 3) as i64);
    let info = lib.model_info("alpha", &hf).await.unwrap().unwrap(); // the real models.config
    assert!(
        info.num_layers == Some(2) && info.weight_bytes == Some(copy.size_bytes),
        "{info:?}"
    );

    // a copy to the other node: the large blob machine to machine (fs.read -> fs.write), the rest from the hub
    let cp = dl
        .copy(
            &copy.id,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("hub".into()),
            },
        )
        .await
        .unwrap();
    dl.wait(&cp.id).await;
    let cp = dl.get(&cp.id).unwrap();
    assert_eq!(cp.state, "completed", "{:?} {:?}", cp.error_code, cp.error);
    let other = b.join("hub").join("models--org--m");
    assert_eq!(
        std::fs::read(other.join("blobs").join(&sha)).unwrap(),
        payload
    );
    assert_eq!(
        std::fs::read(other.join("snapshots").join(COMMIT).join("tok.txt")).unwrap(),
        b"tok"
    );
    let opens = hub.opens();
    assert!(
        !opens
            .iter()
            .rev()
            .take(2)
            .any(|o| o.2 == "model.safetensors"),
        "{opens:?}"
    );
    let l4 = Arc::clone(&lib);
    wait_for("both machines to list it", || {
        l4.models(None, None, Some("org/m"))
            .is_ok_and(|m| m.first().is_some_and(|x| x.copies.len() == 2))
    })
    .await;

    // resume from a real partial, then a corrupted download that the node refuses and a retry that works
    let g = noise(2 * 1024 * 1024 + 7, 2);
    hub.add("org/g", &[("m-Q4_K_M.gguf", &g), ("README.md", b"hi")]);
    let part = a.join("gguf").join("org--g");
    std::fs::create_dir_all(&part).unwrap();
    std::fs::write(part.join("m-Q4_K_M.gguf.incomplete"), &g[..1_000_000]).unwrap();
    let d = dl
        .create(&dreq("org/g", "alpha", "gguf", Some(&["m-Q4_K_M.gguf"])))
        .await
        .unwrap();
    dl.wait(&d.id).await;
    assert_eq!(dl.get(&d.id).unwrap().state, "completed");
    assert_eq!(std::fs::read(part.join("m-Q4_K_M.gguf")).unwrap(), g);
    let offsets: Vec<i64> = hub
        .opens()
        .iter()
        .filter(|o| o.2 == "m-Q4_K_M.gguf")
        .map(|o| o.3)
        .collect();
    assert_eq!(offsets, [1_000_000]);
    hub.st().corrupt.insert("m-Q4_K_M.gguf".into());
    let d = dl
        .create(&dreq("org/g", "beta", "gguf", Some(&["m-Q4_K_M.gguf"])))
        .await
        .unwrap();
    dl.wait(&d.id).await;
    let bad = dl.get(&d.id).unwrap();
    assert!(
        bad.state == "failed" && bad.error_code.as_deref() == Some("hash_mismatch"),
        "{bad:?}"
    );
    assert!(
        bad.error.as_deref().unwrap_or("").starts_with("Beta: "),
        "{bad:?}"
    );
    assert!(incompletes(&b.join("gguf")).is_empty());
    hub.st().corrupt.clear();
    dl.resume(&d.id).await.unwrap();
    dl.wait(&d.id).await;
    assert_eq!(dl.get(&d.id).unwrap().state, "completed");

    // a plain model directory: no listing op exists, so the copy probes names and PROVES it found every file against
    // the byte total and file count the real scanner reported
    let (shard_a, shard_b) = (noise(300_000, 3), noise(120_000, 4));
    let index = json!({"weight_map": {"x": "model-00001-of-00002.safetensors", "y": "model-00002-of-00002.safetensors"}})
        .to_string();
    let plain = a.join("pl").join("mymodel");
    std::fs::create_dir_all(&plain).unwrap();
    for (name, data) in [
        ("config.json", br#"{"model_type": "x"}"#.to_vec()),
        ("tokenizer.json", b"{}".to_vec()),
        ("model.safetensors.index.json", index.into_bytes()),
        ("model-00001-of-00002.safetensors", shard_a),
        ("model-00002-of-00002.safetensors", shard_b.clone()),
    ] {
        std::fs::write(plain.join(name), data).unwrap();
    }
    lib.rescan(Some("alpha")).await.unwrap();
    let pref = mref("plain", "pl", "mymodel");
    let (l5, p5) = (Arc::clone(&lib), pref.clone());
    wait_for("the plain directory", || {
        l5.find_copy("alpha", &p5, false).is_some()
    })
    .await;
    let pcopy = lib.find_copy("alpha", &pref, false).unwrap();
    assert_eq!(pcopy.files, Some(5));
    let pc = dl
        .copy(
            &pcopy.id,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("pl".into()),
            },
        )
        .await
        .unwrap();
    dl.wait(&pc.id).await;
    let pc = dl.get(&pc.id).unwrap();
    assert_eq!(pc.state, "completed", "{:?} {:?}", pc.error_code, pc.error);
    let copied = b.join("pl").join("mymodel");
    assert_eq!(
        std::fs::read(copied.join("model-00002-of-00002.safetensors")).unwrap(),
        shard_b
    );
    assert!(copied.join("config.json").is_file());
    // a file no probe would name: the proof refuses
    std::fs::write(plain.join("surprise.dat"), b"?").unwrap();
    lib.rescan(Some("alpha")).await.unwrap();
    let (l6, p6) = (Arc::clone(&lib), pref.clone());
    wait_for("the rescan", || {
        l6.find_copy("alpha", &p6, false)
            .is_some_and(|c| c.files == Some(6))
    })
    .await;
    let pc2 = dl
        .copy(
            &pcopy.id,
            &CopyRequest {
                host_id: "beta".into(),
                root_id: Some("pl".into()),
            },
        )
        .await
        .unwrap();
    dl.wait(&pc2.id).await;
    let refused = dl.get(&pc2.id).unwrap();
    assert!(
        refused.state == "failed"
            && refused.error_code.as_deref() == Some("copy_listing_incomplete"),
        "{refused:?}"
    );

    // a GGUF file the real scanner read: its header facts become a ModelInfo
    tiny_gguf(&a.join("gguf").join("tiny-Q4_K_M.gguf"));
    lib.rescan(Some("alpha")).await.unwrap();
    let gref = mref("gguf", "gguf", "tiny-Q4_K_M.gguf");
    let (l7, g7) = (Arc::clone(&lib), gref.clone());
    wait_for("the real scanner to read the gguf", || {
        l7.find_copy("alpha", &g7, false).is_some()
    })
    .await;
    let ginfo = lib.model_info("alpha", &gref).await.unwrap().unwrap();
    assert_eq!(
        (
            ginfo.num_layers,
            ginfo.num_attention_heads,
            ginfo.num_kv_heads,
            ginfo.head_dim
        ),
        (Some(2), Some(4), Some(2), Some(16)),
        "{ginfo:?}"
    );
    assert!(ginfo.hidden_size == Some(64) && ginfo.max_position_embeddings == Some(4096));
    assert_eq!(ginfo.quantization.as_deref(), Some("q4_k_m"));
    assert_eq!(ginfo.architectures, ["llama"]);
    assert_eq!(ginfo.num_params, Some(2 * 64 * 16));

    // the node's own refusals reach the caller with their codes
    let e = dl
        .create(&dreq("org/m", "beta", "ro", None))
        .await
        .unwrap_err();
    assert_eq!(e.code(), "root_not_writable");

    // delete one copy through the library: the real fs.delete
    let l8 = Arc::clone(&lib);
    wait_for("beta's copy to be listed", || {
        l8.models(None, None, Some("org/m")).is_ok_and(|m| {
            m.iter()
                .any(|x| x.copies.iter().any(|c| c.host_id == "beta"))
        })
    })
    .await;
    let models = lib.models(None, None, Some("org/m")).unwrap();
    let beta_copy = models[0]
        .copies
        .iter()
        .find(|c| c.host_id == "beta")
        .unwrap()
        .id
        .clone();
    lib.delete_copy(&beta_copy).await.unwrap();
    assert!(!other.exists());
    let hosts: Vec<String> = lib.models(None, None, Some("org/m")).unwrap()[0]
        .copies
        .iter()
        .map(|c| c.host_id.clone())
        .collect();
    assert_eq!(hosts, ["alpha"]);

    // a cancel in the middle of a file removes the partial from the real disk
    let gate = Gate::new();
    {
        let mut st = hub.st();
        st.pass_chunks = Some(2);
        st.gate = Some(gate.clone());
    }
    hub.add("org/big", &[("b.safetensors", &noise(4 * 1024 * 1024, 5))]);
    let d = dl
        .create(&dreq("org/big", "alpha", "hf", None))
        .await
        .unwrap();
    let blobs = a
        .join("hf")
        .join("hub")
        .join("models--org--big")
        .join("blobs");
    let b2 = blobs.clone();
    wait_for("a partial on the real disk", || {
        std::fs::read_dir(&b2).is_ok_and(|rd| {
            rd.flatten().any(|e| {
                e.file_name().to_string_lossy().ends_with(".incomplete")
                    && e.metadata().is_ok_and(|m| m.len() > 0)
            })
        })
    })
    .await;
    dl.cancel(&d.id).await.unwrap();
    assert_eq!(dl.get(&d.id).unwrap().state, "cancelled");
    assert!(incompletes(&blobs).is_empty());

    {
        let mut st = hub.st();
        st.pass_chunks = None;
    }
    gate.set();
    lib.stop().await;
    dl.shutdown().await;
    for l in &links {
        l.stop().await;
    }
}
