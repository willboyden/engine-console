//! The shared test kit of the hub's integration tests (port of backend/tests/core/conftest.py `Env` + fakes.py).
//! Include it from a test file with `#[path = "common/kit.rs"] mod kit;` (no shared mod list:
//! further fakes are new files under tests/common/).
//!
//! `TestEnv::new()` is Python's `Env`: a temp data dir, HF cache and frontend, trust_loopback ON with the client at
//! 127.0.0.1 (so requests are admin without a key, as the Python `client` fixture), Host "testserver" allowed, the
//! fake engine adapter, an in-memory Hugging Face hub and a fake fleet with one machine (`local`, "testbox").

#![allow(dead_code, clippy::panic, clippy::expect_used, unreachable_pub)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::body::Body as AxBody;
use bytes::Bytes;
use ec_hub::adapters::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use ec_hub::app::{AppState, handle_request};
use ec_hub::config::Settings;
use ec_hub::domain::errors::{AppResult, Problem, gated_model, not_found};
use ec_hub::domain::ports::{ByteStream, HubClient, HubModel, RepoFile};
use http_body_util::BodyExt;
use serde_json::{Value, json};

/// A URL on a closed port (9, discard: no lab or CI service listens there); the hermetic default for Hugging Face.
pub const CLOSED_URL: &str = "https://127.0.0.1:9";
/// host:port nothing listens on; the hermetic default for the OTLP endpoint.
pub const CLOSED_HOSTPORT: &str = "127.0.0.1:9";
/// The fake CLIs (docker, nvidia-smi, ip) of the parity harness: put this first on PATH of any child a test starts.
pub const FAKES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/fakes/bin");
/// PATH for children: the fakes, then the system's read-only basics.
pub fn hermetic_path() -> String {
    format!("{FAKES}:/usr/bin:/bin")
}

/// The fake fleet, library and environment (the port of fakefleet.py), as a submodule of the kit: a test includes
/// only this file and reaches them as `kit::fakefleet::...` (`use kit::fakefleet;`) or through the re-exports below.
/// Never include `common/fakefleet.rs` as a module of its own too (clippy `duplicate_mod`; two copies of each type).
#[path = "fakefleet.rs"]
pub mod fakefleet;
#[allow(unused_imports)] // each test binary uses a different subset
pub use fakefleet::{FakeFleet, FakeLibrary, FakeNode, FleetEnv, HelloOpts, make_hello};

pub const GIB: f64 = 1_073_741_824.0;

// ── the environment ──────────────────────────────────────────────────────────────────────────────────────
/// Python's `Env` (tests/core/conftest.py): fakefleet's `FleetEnv` (the app container, the fake fleet with one machine,
/// `local` / "testbox", the fake library and hub, programmable fake engines; trust_loopback ON with the client at
/// 127.0.0.1, Host "testserver" allowed) plus this kit's request helpers (`req`/`get`/`post`/... returning `Req`,
/// with `chunked()` and `open()`, and `admin_key()`). Every field and helper of `FleetEnv` is reachable through
/// `Deref` (`env.node`, `env.fleet`, `env.library`, `env.cfg`, `env.adapters`, `env.add_host(..)`, ...).
pub struct TestEnv(pub FleetEnv);

impl std::ops::Deref for TestEnv {
    type Target = FleetEnv;
    fn deref(&self) -> &FleetEnv {
        &self.0
    }
}

impl std::ops::DerefMut for TestEnv {
    fn deref_mut(&mut self) -> &mut FleetEnv {
        &mut self.0
    }
}

impl TestEnv {
    pub fn new() -> Self {
        Self::with(|_| {})
    }

    /// Python's `Env(tmp)` with `edit` applied to the settings before the container is built. The hermetic defaults
    /// (closed Hugging Face and OTLP endpoints, the fake docker, no node command) are `FleetEnv`'s (PORTING.md §A2).
    pub fn with(edit: impl FnOnce(&mut Settings)) -> Self {
        Self(FleetEnv::with(edit))
    }

    /// The app (built on first use, like the `client` fixture).
    pub fn app(&mut self) -> Arc<AppState> {
        self.0.app()
    }

    /// The app's container (built on first use).
    pub fn container(&mut self) -> Arc<ec_hub::container::Container> {
        self.0.container()
    }

    /// A request builder with the Python `client` fixture's defaults: Host testserver, client 127.0.0.1 and the
    /// `X-Engine-Console: 1` header.
    pub fn req(&mut self, method: &str, path: &str) -> Req {
        Req {
            app: self.app(),
            method: method.into(),
            path: path.into(),
            headers: vec![
                ("host".into(), "testserver".into()),
                ("x-engine-console".into(), "1".into()),
            ],
            body: Vec::new(),
            client: "127.0.0.1".into(),
            no_length: false,
        }
    }

    pub fn get(&mut self, path: &str) -> Req {
        self.req("GET", path)
    }

    pub fn post(&mut self, path: &str) -> Req {
        self.req("POST", path)
    }

    pub fn put(&mut self, path: &str) -> Req {
        self.req("PUT", path)
    }

    pub fn patch(&mut self, path: &str) -> Req {
        self.req("PATCH", path)
    }

    pub fn delete(&mut self, path: &str) -> Req {
        self.req("DELETE", path)
    }

    /// The bootstrap admin key written at start-up (`ensure_bootstrap_key`).
    pub fn admin_key(&mut self) -> String {
        let c = Arc::clone(&self.app().c);
        let data = self.cfg.data_dir.clone();
        c.settings.ensure_bootstrap_key(&data).expect("bootstrap");
        std::fs::read_to_string(data.join("bootstrap-admin.key"))
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    }
}

impl Default for TestEnv {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Req {
    app: Arc<AppState>,
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    client: String,
    no_length: bool,
}

impl Req {
    /// Send the body without a Content-Length (a streamed upload: the cap is enforced while it is read).
    pub fn chunked(mut self) -> Self {
        self.no_length = true;
        self
    }

    pub fn header(mut self, k: &str, v: &str) -> Self {
        let k = k.to_lowercase();
        self.headers.retain(|(kk, _)| *kk != k);
        self.headers.push((k, v.into()));
        self
    }

    pub fn without(mut self, k: &str) -> Self {
        let k = k.to_lowercase();
        self.headers.retain(|(kk, _)| *kk != k);
        self
    }

    pub fn bearer(self, key: &str) -> Self {
        self.header("authorization", &format!("Bearer {key}"))
    }

    pub fn json(mut self, v: &Value) -> Self {
        self.body = serde_json::to_vec(v).expect("json");
        self.header("content-type", "application/json")
    }

    pub fn raw(mut self, b: impl Into<Vec<u8>>) -> Self {
        self.body = b.into();
        self
    }

    /// The caller's address (default 127.0.0.1, the loopback the fixture trusts).
    pub fn client(mut self, host: &str) -> Self {
        self.client = host.into();
        self
    }

    /// Send and return as soon as the response head is there; the body is left to read (streams).
    pub async fn open(self) -> Opened {
        let mut b = http::Request::builder()
            .method(self.method.as_str())
            .uri(self.path.as_str());
        let has_len = self.no_length || self.headers.iter().any(|(k, _)| k == "content-length");
        for (k, v) in &self.headers {
            b = b.header(k.as_str(), v.as_str());
        }
        if !self.body.is_empty() && !has_len {
            b = b.header("content-length", self.body.len().to_string());
        }
        let req = b.body(AxBody::from(self.body)).expect("request");
        let r = handle_request(&self.app, &self.client, req).await;
        let status = r.status().as_u16();
        Opened {
            status,
            body: Some(r.into_body()),
            received: Vec::new(),
        }
    }

    pub async fn send(self) -> Resp {
        let mut b = http::Request::builder()
            .method(self.method.as_str())
            .uri(self.path.as_str());
        let has_len = self.no_length || self.headers.iter().any(|(k, _)| k == "content-length");
        for (k, v) in &self.headers {
            b = b.header(k.as_str(), v.as_str());
        }
        if !self.body.is_empty() && !has_len {
            b = b.header("content-length", self.body.len().to_string());
        }
        let req = b.body(AxBody::from(self.body)).expect("request");
        let r = handle_request(&self.app, &self.client, req).await;
        let status = r.status().as_u16();
        let headers: Vec<(String, String)> = r
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str().to_string(),
                    String::from_utf8_lossy(v.as_bytes()).into_owned(),
                )
            })
            .collect();
        let body = r
            .into_body()
            .collect()
            .await
            .map(|c| c.to_bytes())
            .unwrap_or_default();
        Resp {
            status,
            headers,
            body,
        }
    }
}

/// A response whose body is read on demand (a client that keeps a stream open, or leaves).
pub struct Opened {
    pub status: u16,
    body: Option<AxBody>,
    pub received: Vec<u8>,
}

impl Opened {
    /// The next body chunk, waiting at most `wait`; None at the end, on a timeout, or after `leave`.
    pub async fn chunk(&mut self, wait: Duration) -> Option<Bytes> {
        let body = self.body.as_mut()?;
        match tokio::time::timeout(wait, body.frame()).await {
            Ok(Some(Ok(f))) => {
                let d = f.into_data().ok()?;
                self.received.extend_from_slice(&d);
                Some(d)
            }
            _ => None,
        }
    }

    /// Read to the end (or until `wait` passes with nothing new).
    pub async fn rest(&mut self, wait: Duration) -> Vec<u8> {
        while self.chunk(wait).await.is_some() {}
        self.received.clone()
    }

    /// The client goes away: the body (and whatever it holds, such as a stream slot) is dropped.
    pub fn leave(&mut self) {
        self.body = None;
    }

    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.received).unwrap_or(Value::Null)
    }
}

#[derive(Debug, Clone)]
pub struct Resp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
}

impl Resp {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    pub fn header(&self, k: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(kk, _)| kk.eq_ignore_ascii_case(k))
            .map(|(_, v)| v.as_str())
    }
    /// The problem's `code` member.
    pub fn code(&self) -> String {
        self.json()["code"].as_str().unwrap_or_default().to_string()
    }
}

// ── the fake engine adapter (fakes.py FakeAdapter) ─────────────────────────────────────────────────────────
pub struct FakeAdapter;

fn spec(
    key: &str,
    flag: &str,
    label: &str,
    help: &str,
    typ: &str,
    group: &str,
    mem: bool,
) -> ParamSpec {
    ParamSpec {
        key: key.into(),
        flag: flag.into(),
        label: label.into(),
        help: help.into(),
        type_: typ.into(),
        group: group.into(),
        default: Value::Null,
        choices: None,
        min: None,
        max: None,
        advanced: false,
        affects_memory: mem,
        requires_restart: true,
        bool_style: "flag".into(),
        docs_url: None,
        explain: None,
        level: None,
        unit: None,
    }
}

impl EngineAdapter for FakeAdapter {
    fn id(&self) -> &str {
        "fake"
    }
    fn display_name(&self) -> &str {
        "Fake Engine"
    }
    fn default_image(&self) -> &str {
        "fake/engine:1.0"
    }
    fn param_catalog(&self) -> Vec<ParamSpec> {
        let mut kv = spec("kv_dtype", "--kv", "KV dtype", "kv", "enum", "memory", true);
        kv.choices = Some(vec!["auto".into(), "fp8".into()]);
        vec![
            spec(
                "max_model_len",
                "--max-model-len",
                "Max len",
                "ctx",
                "int",
                "context",
                true,
            ),
            spec(
                "tp",
                "--tp",
                "TP",
                "tensor parallel",
                "int",
                "parallelism",
                true,
            ),
            spec(
                "mem_fraction",
                "--mem",
                "Mem",
                "fraction",
                "float",
                "memory",
                true,
            ),
            kv,
            spec(
                "api_key",
                "env:FAKE_API_KEY",
                "API key",
                "key",
                "string",
                "network",
                false,
            ),
            spec(
                "leaky_key",
                "--leaky-key",
                "Leaky",
                "argv secret",
                "string",
                "network",
                false,
            ),
            spec(
                "trust_remote_code",
                "--trust-remote-code",
                "Trust",
                "t",
                "bool",
                "advanced",
                false,
            ),
            spec(
                "image", "@image", "Image", "override", "string", "advanced", false,
            ),
        ]
    }
    fn presets(&self) -> Vec<(String, Obj)> {
        vec![
            (
                "balanced".into(),
                json!({"mem_fraction": 0.9})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
            (
                "long-context".into(),
                json!({"mem_fraction": 0.9, "max_model_len": 8192})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
        ]
    }
    fn validate(&self, params: &Obj) -> Vec<String> {
        let known: Vec<String> = self.param_catalog().into_iter().map(|p| p.key).collect();
        params
            .keys()
            .filter(|k| !known.contains(k))
            .map(|k| format!("unknown parameter '{k}'"))
            .collect()
    }
    fn build_launch(
        &self,
        model: &str,
        params: &Obj,
        _hw: &Hardware,
        served_name: &str,
        hf: &str,
    ) -> Result<LaunchSpec, Problem> {
        let mut argv: Vec<String> = [
            "--model",
            model,
            "--served-model-name",
            served_name,
            "--port",
            "8000",
        ]
        .map(String::from)
        .to_vec();
        for (k, v) in params {
            if k != "api_key" && k != "image" {
                argv.push(format!("--{}", k.replace('_', "-")));
                argv.push(match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                });
            }
        }
        let key = params
            .get("api_key")
            .and_then(Value::as_str)
            .unwrap_or("s3cr3t-key-value");
        Ok(LaunchSpec {
            image: self.default_image().into(),
            argv,
            env: json!({"HF_HOME": hf, "FAKE_API_KEY": key})
                .as_object()
                .cloned()
                .unwrap_or_default(),
            container_port: 8000,
            health_path: "/health".into(),
            metrics_path: "/metrics".into(),
            models_path: "/v1/models".into(),
            shm_size: "16g".into(),
            ipc_host: true,
            files: Obj::new(),
            needs_hf_cache: true,
            ulimits: Obj::new(),
        })
    }
    fn compatibility(&self, model: &ModelInfo, _p: &Obj, _hw: &Hardware) -> Vec<Compat> {
        if model.quantization.as_deref() == Some("nvfp4") && model.is_moe {
            return vec![Compat {
                level: "block".into(),
                code: "nvfp4_moe_sm120".into(),
                message: "use FP8 or SGLang".into(),
            }];
        }
        Vec::new()
    }
    fn memory_model(&self, _m: &ModelInfo, params: &Obj) -> FloatMap {
        let f = |k: &str, d: f64| params.get(k).and_then(Value::as_f64).unwrap_or(d);
        let mut out = vec![
            ("mem_fraction".to_string(), f("mem_fraction", 0.9)),
            ("tp".to_string(), f("tp", 1.0)),
            (
                "kv_bytes_per_elem".to_string(),
                if params.get("kv_dtype").and_then(Value::as_str) == Some("fp8") {
                    1.0
                } else {
                    2.0
                },
            ),
        ];
        if let Some(m) = params
            .get("max_model_len")
            .and_then(Value::as_f64)
            .filter(|m| *m != 0.0)
        {
            out.push(("max_len".into(), m));
        }
        out
    }
    fn parse_metrics(&self, text: &str) -> FloatMap {
        let map = [
            ("running", "requests_running"),
            ("waiting", "requests_waiting"),
            ("prompt_total", "prompt_tokens_total"),
            ("gen_total", "generation_tokens_total"),
            ("kv", "kv_cache_usage_pct"),
        ];
        let mut out = Vec::new();
        for line in text.lines() {
            let (name, val) = line.split_once(' ').unwrap_or((line, ""));
            if let Some((_, key)) = map.iter().find(|(n, _)| *n == name)
                && let Ok(v) = val.trim().parse::<f64>()
            {
                out.push(((*key).to_string(), v));
            }
        }
        out
    }
    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        if let Some(i) = line.find("loading weights ") {
            let rest = &line[i + 16..];
            let pct: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if !pct.is_empty() && rest[pct.len()..].starts_with('%') {
                return json!({"phase": "loading_weights", "pct": pct.parse::<f64>().unwrap_or(0.0)}).as_object().cloned();
            }
        }
        if line.contains("server ready") {
            return json!({"phase": "ready"}).as_object().cloned();
        }
        None
    }
}

/// fakes.py `FakeMultiAdapter`: a llama.cpp-like engine (gguf, plain and ollama models as well as HF ones, a
/// second image for arm64, its own metric prefixes); the catalogue and the rest are the fake engine's.
pub struct FakeMultiAdapter;

impl EngineAdapter for FakeMultiAdapter {
    fn id(&self) -> &str {
        "fakecpp"
    }
    fn display_name(&self) -> &str {
        "Fake C++ Engine"
    }
    fn default_image(&self) -> &str {
        "fake/engine:1.0"
    }
    fn model_formats(&self) -> Vec<String> {
        ["hf", "gguf", "plain", "ollama"].map(String::from).to_vec()
    }
    fn images_by_arch(&self) -> Vec<(String, String)> {
        vec![("aarch64".into(), "fake/engine:1.0-arm64".into())]
    }
    fn metric_prefixes(&self) -> Option<Vec<String>> {
        Some(vec!["fakecpp:".into()])
    }
    fn build_launch_ctx(
        &self,
        ctx: &ec_hub::adapters::base::LaunchContext,
    ) -> Result<LaunchSpec, Problem> {
        let model = ctx.model_path.clone().unwrap_or_else(|| ctx.model.clone());
        let mut argv: Vec<String> = [
            "--model",
            &model,
            "--alias",
            &ctx.served_name,
            "--port",
            "8000",
        ]
        .map(String::from)
        .to_vec();
        for (k, v) in &ctx.params {
            if k != "api_key" && k != "image" {
                argv.push(format!("--{}", k.replace('_', "-")));
                argv.push(match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                });
            }
        }
        Ok(LaunchSpec {
            image: self.image_for(&ctx.hw.arch),
            argv,
            env: json!({"FAKE_FORMAT": ctx.model_format})
                .as_object()
                .cloned()
                .unwrap_or_default(),
            container_port: 8000,
            health_path: "/health".into(),
            metrics_path: "/metrics".into(),
            models_path: "/v1/models".into(),
            shm_size: "16g".into(),
            ipc_host: true,
            files: Obj::new(),
            needs_hf_cache: ctx.model_format == "hf",
            ulimits: Obj::new(),
        })
    }
    fn param_catalog(&self) -> Vec<ParamSpec> {
        FakeAdapter.param_catalog()
    }
    fn presets(&self) -> Vec<(String, Obj)> {
        FakeAdapter.presets()
    }
    fn validate(&self, params: &Obj) -> Vec<String> {
        FakeAdapter.validate(params)
    }
    fn build_launch(
        &self,
        m: &str,
        p: &Obj,
        hw: &Hardware,
        s: &str,
        h: &str,
    ) -> Result<LaunchSpec, Problem> {
        FakeAdapter.build_launch(m, p, hw, s, h)
    }
    fn compatibility(&self, m: &ModelInfo, p: &Obj, hw: &Hardware) -> Vec<Compat> {
        FakeAdapter.compatibility(m, p, hw)
    }
    fn memory_model(&self, m: &ModelInfo, p: &Obj) -> FloatMap {
        FakeAdapter.memory_model(m, p)
    }
    fn parse_metrics(&self, text: &str) -> FloatMap {
        FakeAdapter.parse_metrics(text)
    }
    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        FakeAdapter.parse_startup_log(line)
    }
}

// ── the fake Hugging Face hub (fakes.py FakeHub) ───────────────────────────────────────────────────────────
/// One repo: its model info and its files (name, bytes).
pub type HubRepo = (HubModel, Vec<(String, Vec<u8>)>);

#[derive(Default)]
pub struct FakeHub {
    pub repos: Mutex<HashMap<String, HubRepo>>,
    pub gated: Mutex<Vec<String>>,
    pub calls: Mutex<Vec<String>>,
}

impl FakeHub {
    pub fn add(
        &self,
        repo_id: &str,
        files: &[(&str, &[u8])],
        config: Option<Value>,
        params: &[(&str, i64)],
    ) {
        use sha2::Digest as _;
        let rf = files
            .iter()
            .map(|(p, b)| RepoFile {
                path: (*p).into(),
                size: i64::try_from(b.len()).unwrap_or(0),
                sha256: p
                    .ends_with(".safetensors")
                    .then(|| hex::encode(sha2::Sha256::digest(b))),
            })
            .collect();
        let total: i64 = params.iter().map(|(_, n)| n).sum();
        let hm = HubModel {
            repo_id: repo_id.into(),
            sha: Some("c0ffee00000000000000000000000000000000ff".into()),
            gated: false,
            license: Some("apache-2.0".into()),
            pipeline_tag: Some("text-generation".into()),
            library_name: Some("transformers".into()),
            tags: Vec::new(),
            files: rf,
            safetensors_params: params.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
            safetensors_total: (total != 0).then_some(total),
            config: config.and_then(|c| c.as_object().cloned()),
            downloads: Some(100),
            likes: Some(5),
        };
        let data = files
            .iter()
            .map(|(p, b)| ((*p).to_string(), b.to_vec()))
            .collect();
        self.repos
            .lock()
            .expect("lock")
            .insert(repo_id.into(), (hm, data));
    }
}

struct FakeByteStream {
    chunks: Vec<Bytes>,
    total: i64,
    offset: i64,
}

#[async_trait]
impl ByteStream for FakeByteStream {
    fn total(&self) -> Option<i64> {
        Some(self.total)
    }
    fn offset(&self) -> i64 {
        self.offset
    }
    async fn chunk(&mut self) -> AppResult<Option<Bytes>> {
        Ok((!self.chunks.is_empty()).then(|| self.chunks.remove(0)))
    }
    async fn close(&mut self) {}
}

#[async_trait]
impl HubClient for FakeHub {
    async fn search(
        &self,
        q: &str,
        _task: Option<&str>,
        _library: Option<&str>,
        _quant: Option<&str>,
        _sort: &str,
        limit: i64,
        _token_ok: bool,
    ) -> AppResult<Vec<HubModel>> {
        self.calls.lock().expect("lock").push(format!("search:{q}"));
        let repos = self.repos.lock().expect("lock");
        let mut hits: Vec<HubModel> = repos
            .values()
            .filter(|(m, _)| m.repo_id.to_lowercase().contains(&q.to_lowercase()))
            .map(|(m, _)| m.clone())
            .collect();
        hits.truncate(usize::try_from(limit).unwrap_or(0));
        Ok(hits)
    }
    async fn model(&self, repo_id: &str, _revision: Option<&str>) -> AppResult<HubModel> {
        self.calls
            .lock()
            .expect("lock")
            .push(format!("model:{repo_id}"));
        if self
            .gated
            .lock()
            .expect("lock")
            .iter()
            .any(|g| g == repo_id)
        {
            return Err(gated_model(format!("gated: {repo_id}")).into());
        }
        self.repos
            .lock()
            .expect("lock")
            .get(repo_id)
            .map(|(m, _)| m.clone())
            .ok_or_else(|| not_found(format!("not found on Hugging Face: {repo_id}")).into())
    }
    async fn card(&self, repo_id: &str, _revision: Option<&str>) -> AppResult<String> {
        Ok(format!("# {repo_id}\nA model card."))
    }
    async fn open_file(
        &self,
        repo_id: &str,
        _revision: &str,
        path: &str,
        start: i64,
    ) -> AppResult<Box<dyn ByteStream>> {
        let repos = self.repos.lock().expect("lock");
        let data = repos
            .get(repo_id)
            .and_then(|(_, f)| f.iter().find(|(p, _)| p == path))
            .map(|(_, b)| b.clone())
            .ok_or_else(|| not_found(format!("not found on Hugging Face: {repo_id}")))?;
        let s = usize::try_from(start).unwrap_or(0).min(data.len());
        let chunks = data[s..].chunks(4).map(Bytes::copy_from_slice).collect();
        Ok(Box::new(FakeByteStream {
            chunks,
            total: i64::try_from(data.len()).unwrap_or(0),
            offset: start,
        }))
    }
}

/// The temp path as a String (for assertions on messages that name files).
pub fn path_str(p: &std::path::Path) -> String {
    p.to_string_lossy().into_owned()
}

pub fn pb(p: &str) -> PathBuf {
    PathBuf::from(p)
}
