//! Port of backend/tests/data/test_routers.py: the data routes over HTTP (/library, /downloads, /models, /hf/models and
//! PUT /settings) with the real error handlers and role checks.
//!
//! Python mounts the routers on a small app with a FakeFit and sets the caller's role directly. Here the routes are
//! the real app's (core's composition root over the data world's store, fleet and hub, with the world's library,
//! downloads and settings), a viewer is a remote caller with a viewer key, and `/models` is judged by the real fit
//! estimator: the local machine has a GPU, Spark has none, and of the stub engines only "vllm" has a memory model
//! that fits (the others' `mem_fraction` is tiny) - the verdicts Python's FakeFit hands out.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/wsc_data.rs"]
mod wsc_data;

use std::path::Path;
use std::sync::Arc;

use ec_hub::adapters::AdapterRegistry;
use ec_hub::adapters::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use ec_hub::adapters::pycompat::raise;
use ec_hub::app::{AppState, app_state, handle_request};
use ec_hub::container::{Container, Overrides, build_container};
use ec_hub::domain::errors::Problem;
use ec_hub::domain::fleet::Fleet;
use ec_hub::domain::ports::HubClient;
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use wsc_data::*;

const P: &str = "/api/v1";

/// A stub engine whose memory model raises (Python's FakeFit raises for engine "boom").
struct Boom;

impl EngineAdapter for Boom {
    fn id(&self) -> &str {
        "boom"
    }
    fn display_name(&self) -> &str {
        "boom"
    }
    fn default_image(&self) -> &str {
        "fake/boom:1"
    }
    fn model_formats(&self) -> Vec<String> {
        vec!["hf".into()]
    }
    fn param_catalog(&self) -> Vec<ParamSpec> {
        Vec::new()
    }
    fn presets(&self) -> Vec<(String, Obj)> {
        Vec::new()
    }
    fn validate(&self, _params: &Obj) -> Vec<String> {
        Vec::new()
    }
    fn build_launch(
        &self,
        _m: &str,
        _p: &Obj,
        _hw: &Hardware,
        _s: &str,
        _h: &str,
    ) -> Result<LaunchSpec, Problem> {
        Err(ec_hub::domain::errors::internal("boom"))
    }
    fn compatibility(&self, _m: &ModelInfo, _p: &Obj, _hw: &Hardware) -> Vec<Compat> {
        Vec::new()
    }
    fn memory_model(&self, _m: &ModelInfo, _p: &Obj) -> FloatMap {
        raise("RuntimeError", "one odd model")
    }
    fn parse_metrics(&self, _t: &str) -> FloatMap {
        Vec::new()
    }
    fn parse_startup_log(&self, _l: &str) -> Option<Obj> {
        None
    }
}

/// The engines /models judges: only vllm's estimate can fit.
fn engines(boom: bool) -> AdapterRegistry {
    let tiny = [("mem_fraction", 1e-9)];
    let mut list: Vec<Arc<dyn EngineAdapter>> = vec![
        Arc::new(StubAdapter::new("vllm", &["hf", "gguf"])),
        Arc::new(StubAdapter::new("llamacpp", &["gguf", "plain"]).with_mem(&tiny)),
        Arc::new(StubAdapter::new("ollama", &["ollama"]).with_mem(&tiny)),
    ];
    if boom {
        list.push(Arc::new(Boom));
    }
    AdapterRegistry::new(list)
}

struct Env {
    w: World,
    st: Arc<AppState>,
    /// the viewer's key (None: the loopback admin)
    viewer: Option<String>,
}

struct Res {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Res {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    fn header(&self, k: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(kk, _)| kk.eq_ignore_ascii_case(k))
            .map(|(_, v)| v.as_str())
    }
    /// test_routers.problem: a problem+json body whose status is the HTTP status, with a code and a detail.
    fn problem(&self) -> Value {
        assert!(
            self.header("content-type")
                .unwrap_or("")
                .starts_with("application/problem+json"),
            "{}",
            self.text()
        );
        let b = self.json();
        assert_eq!(b["status"], json!(self.status));
        assert!(
            b["code"].as_str().is_some_and(|c| !c.is_empty())
                && b["detail"].as_str().is_some_and(|d| !d.is_empty())
        );
        b
    }
    fn code(&self) -> String {
        self.problem()["code"].as_str().unwrap().to_string()
    }
}

impl Env {
    async fn call(&self, method: &str, path: &str, body: Option<&Value>) -> Res {
        let mut b = http::Request::builder()
            .method(method)
            .uri(path)
            .header("host", "testserver")
            .header("x-engine-console", "1");
        let client = match &self.viewer {
            Some(k) => {
                b = b.header("authorization", format!("Bearer {k}"));
                "192.0.2.7"
            }
            None => "127.0.0.1",
        };
        let bytes = match body {
            Some(v) => {
                b = b.header("content-type", "application/json");
                serde_json::to_vec(v).unwrap()
            }
            None => Vec::new(),
        };
        if !bytes.is_empty() {
            b = b.header("content-length", bytes.len().to_string());
        }
        let req = b.body(axum::body::Body::from(bytes)).unwrap();
        let r = handle_request(&self.st, client, req).await;
        let status = r.status().as_u16();
        let headers = r
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    String::from_utf8_lossy(v.as_bytes()).into_owned(),
                )
            })
            .collect();
        let body = r.into_body().collect().await.unwrap().to_bytes().to_vec();
        Res {
            status,
            headers,
            body,
        }
    }
    async fn get(&self, path: &str) -> Res {
        self.call("GET", path, None).await
    }
    async fn post(&self, path: &str, body: Option<&Value>) -> Res {
        self.call("POST", path, body).await
    }
    async fn delete(&self, path: &str) -> Res {
        self.call("DELETE", path, None).await
    }
    async fn put(&self, path: &str, body: &Value) -> Res {
        self.call("PUT", path, Some(body)).await
    }
    /// The first copy of the model whose name is `q`.
    async fn copy_of(&self, q: &str) -> Value {
        self.get(&format!("{P}/library?q={q}")).await.json()["items"][0]["copies"][0].clone()
    }
}

/// The data world's settings plus what the HTTP layer checks: Host "testserver", loopback trusted (admin without a
/// key, as Python's role shim), and every outbound default closed (PORTING.md §A2).
fn app_settings(tmp: &Path) -> ec_hub::config::Settings {
    let mut s = make_settings(tmp);
    s.allowed_hosts = vec!["testserver".into()];
    s.trust_loopback = true;
    s.hosts_file = tmp.join("hosts.json");
    s.hub_key_file = tmp.join("ssh").join("hub-key");
    s.otlp_endpoint = "127.0.0.1:9".into();
    s.egress_proxy = None;
    s.docker_bin = "/nonexistent/docker".into();
    s.node_policy = None;
    s.node_command = None;
    s
}

async fn env_with(tmp: &Path, role: &str, boom: bool) -> Env {
    let local = node(
        tmp,
        "local",
        NodeOpts {
            label: Some("This machine".into()),
            roots: vec![
                RootCfg::new("hf", "hf-home"),
                RootCfg::new("ro", "plain").ro(),
            ],
            addresses: vec!["10.0.0.1".into()],
            gpus: 1,
            ..NodeOpts::default()
        },
    );
    let other = node(
        tmp,
        "spark",
        NodeOpts {
            label: Some("Spark".into()),
            roots: vec![RootCfg::new("hf", "hf-hub")],
            addresses: vec!["10.0.0.2".into()],
            ..NodeOpts::default()
        },
    );
    seed_hf_cache(
        &local.root_path("hf"),
        "org/m",
        &files(),
        "hf-home",
        COMMIT,
        "main",
    );
    // what the node's models.config answers for it: a real (tiny) geometry, so the fit estimate has something to work
    // from (Python's FakeFit needs none)
    local.st().configs.insert(
        ("hf".into(), "hub/models--org--m".into()),
        json!({"model_type": "llama", "architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2,
               "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
               "max_position_embeddings": 4096, "torch_dtype": "bfloat16"}),
    );
    local.scan();
    seed_hf_cache(
        &other.root_path("hf"),
        "org/s",
        &[("w.safetensors".to_string(), vec![b's'; 40])],
        "hf-hub",
        COMMIT,
        "main",
    );
    other.scan();
    let w = World::new(tmp, vec![local, other], WorldOpts::default())
        .start()
        .await;
    w.hub.add_files("org/m", &files(), None);
    w.hub.add_files("org/new", &files(), None);
    let built = build_container(
        Arc::new(app_settings(tmp)),
        Overrides {
            adapters: Some(engines(boom)),
            fleet: Some(Arc::clone(&w.fleet) as Arc<dyn Fleet>),
            hub: Some(Arc::new(w.hub.clone()) as Arc<dyn HubClient>),
            store: Some(Arc::clone(&w.store)),
            frontend: Some(None),
            ..Overrides::default()
        },
    )
    .expect("container");
    // the routes reach the data world's own services
    let c = Arc::new(Container {
        cfg: Arc::clone(&built.cfg),
        store: Arc::clone(&built.store),
        bus: Arc::clone(&w.bus),
        adapters: built.adapters.clone(),
        settings: Arc::clone(&w.settings),
        fleet: Arc::clone(&built.fleet),
        hub: Arc::clone(&built.hub),
        hardware: Arc::clone(&built.hardware),
        hf: Arc::clone(&built.hf),
        library: Arc::clone(&w.library),
        downloads: Arc::clone(&w.downloads),
        fit: Arc::clone(&built.fit),
        scripts: Arc::clone(&built.scripts),
        nodehttp: Arc::clone(&built.nodehttp),
        lifecycle: Arc::clone(&built.lifecycle),
        profiles: Arc::clone(&built.profiles),
        metrics: Arc::clone(&built.metrics),
        bench: Arc::clone(&built.bench),
        usage: Arc::clone(&built.usage),
        chat: Arc::clone(&built.chat),
        audit: Arc::clone(&built.audit),
        telemetry: Arc::clone(&built.telemetry),
        sse: Arc::clone(&built.sse),
        chat_streams: Arc::clone(&built.chat_streams),
        frontend: None,
        learn: None,
    });
    let viewer = (role == "viewer").then(|| {
        w.settings
            .create_key("v", "viewer")
            .unwrap()
            .secret
            .unwrap()
    });
    Env {
        w,
        st: app_state(c),
        viewer,
    }
}

async fn env(tmp: &Path, role: &str) -> Env {
    env_with(tmp, role, false).await
}

// ---- GET /library ---------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_get_library_lists_models_with_copies() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    let r = e.get(&format!("{P}/library")).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let body = r.json();
    assert!(body["next_cursor"].is_null());
    assert_eq!(
        body["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["key"].clone())
            .collect::<Vec<_>>(),
        [json!("hf:org/m"), json!("hf:org/s")]
    );
    let m = &body["items"][0];
    assert!(
        m["format"] == "hf" && m["engines"] == json!(["vllm"]) && m["size_bytes"] == TOTAL,
        "{m}"
    );
    let c = &m["copies"][0];
    assert!(c["host_id"] == "local" && c["host_label"] == "This machine" && c["root_id"] == "hf");
    assert_eq!(c["visible_on"], json!(["local"]));
    assert!(c["writable"] == true && c["deletable"] == true && c["in_use_by"] == json!([]));
    assert_eq!(c["drive"]["mount"], "/");
    assert_eq!(c["id"].as_str().unwrap().len(), 16);
    assert!(c["path"].as_str().unwrap().ends_with("hub/models--org--m"));
    e.w.stop().await;
}

#[tokio::test]
async fn test_get_library_filters_and_paginates() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    let names = |v: Value| -> Vec<Value> {
        v["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].clone())
            .collect()
    };
    assert_eq!(
        names(e.get(&format!("{P}/library?host=spark")).await.json()),
        [json!("org/s")]
    );
    assert_eq!(
        names(e.get(&format!("{P}/library?q=org/m")).await.json()),
        [json!("org/m")]
    );
    assert_eq!(
        e.get(&format!("{P}/library?format=gguf")).await.json()["items"],
        json!([])
    );
    let page = e.get(&format!("{P}/library?limit=1")).await.json();
    assert!(page["items"].as_array().unwrap().len() == 1 && page["next_cursor"] == "1");
    let rest = e.get(&format!("{P}/library?limit=1&cursor=1")).await.json();
    assert!(rest["next_cursor"].is_null());
    assert_eq!(names(rest), [json!("org/s")]);
    let long = "x".repeat(201);
    for bad in [
        "host=Bad_Host",
        "host=-x",
        "format=nope",
        &format!("q={long}"),
    ] {
        let r = e.get(&format!("{P}/library?{bad}")).await;
        assert!(
            r.status == 422 && r.code() == "validation_error",
            "{bad}: {}",
            r.text()
        );
    }
    e.w.stop().await;
}

#[tokio::test]
async fn test_a_viewer_can_read_the_library() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "viewer").await;
    assert_eq!(e.get(&format!("{P}/library")).await.status, 200);
    assert_eq!(e.get(&format!("{P}/models")).await.status, 200);
    e.w.stop().await;
}

// ---- rescan, delete, copy -------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_rescan() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    let r = e.post(&format!("{P}/library/rescan"), None).await;
    assert!(
        r.status == 202 && r.json() == json!({"hosts": ["local", "spark"]}),
        "{}",
        r.text()
    );
    let r = e
        .post(
            &format!("{P}/library/rescan"),
            Some(&json!({"host_id": "spark"})),
        )
        .await;
    assert!(r.status == 202 && r.json() == json!({"hosts": ["spark"]}));
    let r = e
        .post(
            &format!("{P}/library/rescan"),
            Some(&json!({"host_id": "nobody"})),
        )
        .await;
    assert!(r.status == 404 && r.code() == "unknown_host");
    let r = e
        .post(
            &format!("{P}/library/rescan"),
            Some(&json!({"host_id": "Bad Id"})),
        )
        .await;
    assert_eq!(r.status, 422);
    e.w.fleet.fake("spark").st().state = "down".into();
    let r = e
        .post(
            &format!("{P}/library/rescan"),
            Some(&json!({"host_id": "spark"})),
        )
        .await;
    let p = r.problem();
    assert!(r.status == 503 && p["code"] == "node_unavailable", "{p}");
    assert!(p["detail"].as_str().unwrap().starts_with("Spark: "), "{p}");
    e.w.stop().await;
}

#[tokio::test]
async fn test_delete_a_copy() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    let copy = e.copy_of("org/m").await;
    let cid = copy["id"].as_str().unwrap().to_string();
    e.w.set_busy("local", copy["path"].as_str().unwrap(), &["inst_1"]);
    let r = e.delete(&format!("{P}/library/copies/{cid}")).await;
    let p = r.problem();
    assert!(
        r.status == 409
            && p["code"] == "model_in_use"
            && p["detail"].as_str().unwrap().contains("inst_1"),
        "{p}"
    );
    e.w.clear_busy();
    e.w.fleet.fake("local").st().allow_delete = false;
    let r = e.delete(&format!("{P}/library/copies/{cid}")).await;
    let p = r.problem();
    assert!(r.status == 403 && p["code"] == "policy_denied", "{p}");
    assert!(
        p["detail"].as_str().unwrap().starts_with("This machine: "),
        "{p}"
    );
    e.w.fleet.fake("local").st().allow_delete = true;
    let r = e.delete(&format!("{P}/library/copies/{cid}")).await;
    assert!(r.status == 204 && r.body.is_empty(), "{}", r.text());
    let left: Vec<Value> = e.get(&format!("{P}/library")).await.json()["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].clone())
        .collect();
    assert_eq!(left, [json!("org/s")]);
    let r = e.delete(&format!("{P}/library/copies/{cid}")).await;
    assert!(r.status == 404 && r.code() == "copy_not_found");
    assert_eq!(
        e.delete(&format!("{P}/library/copies/not-an-id"))
            .await
            .status,
        404
    );
    e.w.stop().await;
}

#[tokio::test]
async fn test_copy_a_copy() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    let cid = e.copy_of("org/m").await["id"].as_str().unwrap().to_string();
    let r = e
        .post(
            &format!("{P}/library/copies/{cid}/copy"),
            Some(&json!({"host_id": "spark", "root_id": "hf"})),
        )
        .await;
    assert_eq!(r.status, 201, "{}", r.text());
    let d = r.json();
    assert!(d["kind"] == "copy" && d["source_copy_id"] == json!(cid) && d["host_id"] == "spark");
    assert!(d["root_id"] == "hf" && d["repo_id"] == "org/m");
    let did = d["id"].as_str().unwrap().to_string();
    e.w.downloads.wait(&did).await;
    let got = e.get(&format!("{P}/downloads/{did}")).await.json();
    assert!(
        got["state"] == "completed" && got["total_bytes"] == TOTAL,
        "{got}"
    );
    let r = e
        .post(
            &format!("{P}/library/copies/{cid}/copy"),
            Some(&json!({"host_id": "local", "root_id": "hf"})),
        )
        .await;
    assert!(r.status == 409 && r.code() == "copy_same_location");
    let r = e
        .post(
            &format!("{P}/library/copies/{cid}/copy"),
            Some(&json!({"host_id": "Bad Id"})),
        )
        .await;
    assert_eq!(r.status, 422);
    let zeros = "0".repeat(16);
    let r = e
        .post(
            &format!("{P}/library/copies/{zeros}/copy"),
            Some(&json!({"host_id": "spark"})),
        )
        .await;
    assert!(r.status == 404 && r.code() == "copy_not_found");
    e.w.stop().await;
}

/// The five parametrized requests of `test_a_viewer_cannot_change_anything`.
#[tokio::test]
async fn test_a_viewer_cannot_change_anything() {
    let zeros = "0".repeat(16);
    let cases: [(&str, String, Option<Value>); 5] = [
        ("POST", "/library/rescan".into(), None),
        ("DELETE", format!("/library/copies/{zeros}"), None),
        (
            "POST",
            format!("/library/copies/{zeros}/copy"),
            Some(json!({"host_id": "spark"})),
        ),
        ("DELETE", "/models/org/m".into(), None),
        ("PUT", "/settings".into(), Some(json!({"idle_ttl_s": 5}))),
    ];
    for (method, path, body) in cases {
        let t = tempfile::tempdir().unwrap();
        let e = env(t.path(), "viewer").await;
        let r = e.call(method, &format!("{P}{path}"), body.as_ref()).await;
        assert!(
            r.status == 403 && r.code() == "forbidden",
            "{method} {path}: {}",
            r.text()
        );
        e.w.stop().await;
    }
}

// ---- /downloads -----------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_download_to_a_machine_and_a_root() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    let r = e
        .post(
            &format!("{P}/downloads"),
            Some(&json!({"repo_id": "org/new", "host_id": "spark", "root_id": "hf"})),
        )
        .await;
    assert_eq!(r.status, 201, "{}", r.text());
    let d = r.json();
    assert_eq!(
        (
            &d["host_id"],
            &d["root_id"],
            &d["kind"],
            &d["repo_id"],
            &d["source_copy_id"]
        ),
        (
            &json!("spark"),
            &json!("hf"),
            &json!("hf"),
            &json!("org/new"),
            &Value::Null
        )
    );
    let did = d["id"].as_str().unwrap().to_string();
    e.w.downloads.wait(&did).await;
    let done = e.get(&format!("{P}/downloads/{did}")).await.json();
    assert!(
        done["state"] == "completed" && done["host_id"] == "spark",
        "{done}"
    );
    let ids: Vec<Value> = e.get(&format!("{P}/downloads")).await.json()["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["id"].clone())
        .collect();
    assert_eq!(ids, [json!(did)]);
    let snap = e.get(&format!("{P}/downloads/stream?once=true")).await;
    let text = snap.text();
    assert!(
        snap.status == 200 && text.contains("event: snapshot"),
        "{text}"
    );
    let data: Value =
        serde_json::from_str(text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap()).unwrap();
    assert!(
        data[0]["host_id"] == "spark" && data[0]["kind"] == "hf",
        "{data}"
    );
    assert_eq!(e.delete(&format!("{P}/downloads/{did}")).await.status, 204);
    assert_eq!(e.get(&format!("{P}/downloads/{did}")).await.status, 404);
    e.w.stop().await;
}

#[tokio::test]
async fn test_download_defaults_to_the_local_machine_and_validates_input() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    let r = e
        .post(
            &format!("{P}/downloads"),
            Some(&json!({"repo_id": "org/new"})),
        )
        .await;
    assert!(
        r.status == 201 && r.json()["host_id"] == "local" && r.json()["root_id"] == "hf",
        "{}",
        r.text()
    );
    e.w.downloads.wait(r.json()["id"].as_str().unwrap()).await;
    for (body, status, code) in [
        (json!({"repo_id": "../etc"}), 422, "validation_error"),
        (
            json!({"repo_id": "a/b", "host_id": "BAD"}),
            422,
            "validation_error",
        ),
        (
            json!({"repo_id": "a/b", "host_id": "nobody"}),
            404,
            "unknown_host",
        ),
        (
            json!({"repo_id": "a/b", "root_id": "ro"}),
            409,
            "root_not_writable",
        ),
        (
            json!({"repo_id": "a/b", "root_id": "zzz"}),
            422,
            "unknown_root",
        ),
    ] {
        let r = e.post(&format!("{P}/downloads"), Some(&body)).await;
        assert!(
            r.status == status && r.code() == code,
            "{body}: {}",
            r.text()
        );
    }
    e.w.stop().await;
}

#[tokio::test]
async fn test_pause_resume_cancel_routes() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    let gate = Gate::new();
    e.w.hub.st().gate = Some(gate.clone());
    let d = e
        .post(
            &format!("{P}/downloads"),
            Some(&json!({"repo_id": "org/new", "host_id": "spark"})),
        )
        .await
        .json();
    let did = d["id"].as_str().unwrap().to_string();
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert_eq!(
        e.post(&format!("{P}/downloads/{did}/pause"), None)
            .await
            .json()["state"],
        "paused"
    );
    assert_eq!(
        e.post(&format!("{P}/downloads/{did}/pause"), None)
            .await
            .status,
        409
    );
    gate.set();
    let st = e
        .post(&format!("{P}/downloads/{did}/resume"), None)
        .await
        .json()["state"]
        .clone();
    assert!(st == "queued" || st == "running", "{st}");
    e.w.downloads.wait(&did).await;
    // already completed
    assert_eq!(
        e.post(&format!("{P}/downloads/{did}/cancel"), None)
            .await
            .status,
        409
    );
    e.w.stop().await;
}

// ---- GET /hf/models/{repo}: gguf_groups arrive without the route being touched -------------------------------------
#[tokio::test]
async fn test_the_hf_model_route_returns_the_gguf_groups() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    e.w.hub.add(
        "org/g-GGUF",
        &[
            ("g-Q4_K_M.gguf", b"1234"),
            ("g-Q8_0-00001-of-00002.gguf", b"12345"),
            ("g-Q8_0-00002-of-00002.gguf", b"678"),
            ("README.md", b"x"),
        ],
    );
    let r = e.get(&format!("{P}/hf/models/org/g-GGUF")).await;
    assert_eq!(r.status, 200, "{}", r.text());
    assert_eq!(
        r.json()["gguf_groups"],
        json!([{"quant": "Q4_K_M", "files": ["g-Q4_K_M.gguf"], "total_bytes": 4},
               {"quant": "Q8_0", "files": ["g-Q8_0-00001-of-00002.gguf", "g-Q8_0-00002-of-00002.gguf"],
                "total_bytes": 8}])
    );
    assert_eq!(
        e.get(&format!("{P}/hf/models/org/m")).await.json()["gguf_groups"],
        json!([])
    );
    e.w.stop().await;
}

// ---- /models (v1 shape) ---------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_models_keeps_the_v1_shape_with_a_host_on_every_row() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    let r = e.get(&format!("{P}/models")).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let items = r.json()["items"].as_array().unwrap().clone();
    assert_eq!(
        items
            .iter()
            .map(|m| (m["repo_id"].clone(), m["host_id"].clone()))
            .collect::<Vec<_>>(),
        [
            (json!("org/m"), json!("local")),
            (json!("org/s"), json!("spark"))
        ]
    );
    let mut keys: Vec<&String> = items[0].as_object().unwrap().keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "engines_that_fit",
            "host_id",
            "last_used",
            "path",
            "repo_id",
            "revisions",
            "size_bytes"
        ]
    );
    assert!(items[0]["revisions"] == json!([COMMIT]) && items[0]["size_bytes"] == TOTAL);
    assert_eq!(items[0]["last_used"], json!(2.0));
    // judged on its own machine: the local one has a GPU, Spark has none
    assert_eq!(items[0]["engines_that_fit"], json!(["vllm"]));
    assert_eq!(items[1]["engines_that_fit"], json!([]));
    e.w.stop().await;
}

#[tokio::test]
async fn test_one_model_that_cannot_be_judged_does_not_blank_the_list() {
    ec_hub::adapters::pycompat::install_quiet_panic_hook();
    let t = tempfile::tempdir().unwrap();
    let e = env_with(t.path(), "admin", true).await;
    let r = e.get(&format!("{P}/models")).await;
    assert_eq!(r.status, 200, "{}", r.text());
    assert_eq!(r.json()["items"][0]["engines_that_fit"], json!(["vllm"]));
    e.w.stop().await;
}

#[tokio::test]
async fn test_models_pages() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    let page = e.get(&format!("{P}/models?limit=1")).await.json();
    assert_eq!(page["items"][0]["repo_id"], "org/m");
    assert_eq!(page["items"].as_array().unwrap().len(), 1);
    assert_eq!(page["next_cursor"], "1");
    e.w.stop().await;
}

#[tokio::test]
async fn test_delete_model_removes_this_machines_cache_only() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    // only spark has it
    let r = e.delete(&format!("{P}/models/org/s")).await;
    assert!(r.status == 404 && r.code() == "not_found", "{}", r.text());
    let local = e.copy_of("org/m").await;
    e.w.set_busy("local", local["path"].as_str().unwrap(), &["inst_1"]);
    let r = e.delete(&format!("{P}/models/org/m")).await;
    assert!(r.status == 409 && r.code() == "model_in_use");
    e.w.clear_busy();
    assert_eq!(e.delete(&format!("{P}/models/org/m")).await.status, 204);
    let left: Vec<Value> = e.get(&format!("{P}/models")).await.json()["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["repo_id"].clone())
        .collect();
    assert_eq!(left, [json!("org/s")]);
    assert_eq!(
        e.delete(&format!("{P}/models/org/m")).await.code(),
        "not_found"
    );
    let s = e.delete(&format!("{P}/models/..%2Fetc")).await.status;
    assert!(s == 404 || s == 422, "{s}");
    // the repo id rule, as in v1
    assert_eq!(e.delete(&format!("{P}/models/a--b/c")).await.status, 422);
    e.w.stop().await;
}

// ---- PUT /settings: host_image_pins ---------------------------------------------------------------------------------
#[tokio::test]
async fn test_put_settings_host_image_pins() {
    let t = tempfile::tempdir().unwrap();
    let e = env(t.path(), "admin").await;
    assert_eq!(
        e.get(&format!("{P}/settings")).await.json()["host_image_pins"],
        json!({})
    );
    let pins = json!({"spark": {"vllm": "vllm/vllm-openai:v0.12.0-arm64"}, "local": {"sglang": "lmsysorg/sglang:v0.5.1"}});
    let r = e
        .put(&format!("{P}/settings"), &json!({"host_image_pins": pins}))
        .await;
    assert!(
        r.status == 200 && r.json()["host_image_pins"] == pins,
        "{}",
        r.text()
    );
    assert_eq!(
        e.get(&format!("{P}/settings")).await.json()["host_image_pins"],
        pins
    );
    assert_eq!(
        e.w.settings.image_pin("vllm", "spark").unwrap().as_deref(),
        Some("vllm/vllm-openai:v0.12.0-arm64")
    );
    assert!(e.w.settings.image_pin("vllm", "local").unwrap().is_none());
    // another setting leaves the pins alone
    let r = e
        .put(&format!("{P}/settings"), &json!({"idle_ttl_s": 30}))
        .await;
    assert!(r.json()["host_image_pins"] == pins && r.json()["idle_ttl_s"] == 30);
    let r = e
        .put(&format!("{P}/settings"), &json!({"host_image_pins": {}}))
        .await;
    assert_eq!(r.json()["host_image_pins"], json!({}));
    for bad in [
        json!({"Spark": {"vllm": "a/b:1"}}),
        json!({"spark": {"vllm": "a/b:latest"}}),
        json!({"spark": {"vllm": "a/b"}}),
        json!({"spark": {"vllm": "a/b:1 --privileged"}}),
        json!({"spark": {"bad engine": "a/b:1"}}),
    ] {
        let r = e
            .put(&format!("{P}/settings"), &json!({"host_image_pins": bad}))
            .await;
        assert!(
            r.status == 400 && r.code() == "invalid_setting",
            "{bad}: {}",
            r.text()
        );
    }
    let r = e
        .put(
            &format!("{P}/settings"),
            &json!({"host_image_pins": {"spark": "not a map"}}),
        )
        .await;
    assert_eq!(r.status, 422);
    // a refused change changed nothing
    assert_eq!(
        e.get(&format!("{P}/settings")).await.json()["host_image_pins"],
        json!({})
    );
    e.w.stop().await;
}

/// The library half of test_library_lifecycle_wiring.py's L2 row (the lifecycle tests port that row over the fake
/// library): the real
/// LibraryService of the composition root asks the lifecycle which instances use a copy while the lifecycle asks the
/// same library where its model lives. Neither may hold a lock across the other's call: the copy lists the instance,
/// the instance has a model location, and nothing hangs (a deadlock would fail the timeout, not hang the suite).
#[tokio::test]
async fn test_wsc_l2_library_and_lifecycle_read_each_other_without_recursing() {
    let t = tempfile::tempdir().unwrap();
    let local = node(
        t.path(),
        "local",
        NodeOpts {
            label: Some("L".into()),
            roots: vec![RootCfg::new("hf", "hf-home")],
            gpus: 1,
            ..NodeOpts::default()
        },
    );
    seed_hf_cache(
        &local.root_path("hf"),
        "org/m",
        &files(),
        "hf-home",
        COMMIT,
        "main",
    );
    local.scan();
    let fleet = Arc::new(DataFleet::new(
        Arc::new(ec_hub::services::common::EventBus::new()),
        vec![local],
    ));
    let c = build_container(
        Arc::new(app_settings(t.path())),
        Overrides {
            adapters: Some(engines(false)),
            fleet: Some(fleet as Arc<dyn Fleet>),
            hub: Some(Arc::new(DataHub::new()) as Arc<dyn HubClient>),
            frontend: Some(None),
            ..Overrides::default()
        },
    )
    .expect("container");
    c.library.refresh("local").await.unwrap();
    c.store
        .execute(
            "INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,port,container_name,image,state,\
             created_at,host_id,model) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            &[
                &"inst_aaaaaaaaaaaa",
                &"m",
                &"vllm",
                &"org/m",
                &"{}",
                &"[0]",
                &"[]",
                &18001,
                &"ec-m-aaaaaa",
                &"vllm/vllm-openai:v1",
                &"ready",
                &ec_hub::services::common::now(),
                &"local",
                &r#"{"format": "hf", "repo_id": "org/m"}"#,
            ],
        )
        .unwrap();
    let lib = Arc::clone(&c.library);
    let models = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::task::spawn_blocking(move || lib.models(None, None, None)),
    )
    .await
    .expect("the library and the lifecycle deadlocked")
    .unwrap()
    .unwrap();
    let in_use: Vec<Vec<String>> = models
        .iter()
        .flat_map(|m| m.copies.iter().map(|c| c.in_use_by.clone()))
        .collect();
    assert_eq!(in_use, [vec!["inst_aaaaaaaaaaaa".to_string()]]);
    assert!(
        c.lifecycle
            .get("inst_aaaaaaaaaaaa")
            .unwrap()
            .model_location
            .is_some()
    );
}
