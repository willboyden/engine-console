//! Python: backend/src/engine_console/services/downloads.py
//! (tests/data/test_downloads.py, test_copy.py).
//!
//! Downloads and copies into any machine's model roots (contract v2, docs/ARCHITECTURE.md 10.6). Bytes never touch
//! the hub's own disk and the hub never opens a model directory: it streams them.
//!
//! * **Download** (`kind="hf"`): Hugging Face -> hub (allowlisted egress, optional proxy, the HF token stays here) ->
//!   the target machine's node (`fs.write`, which hashes the whole file and only then renames it into place).
//! * **Copy** (`kind="copy"`): a library copy's machine (`fs.read`) -> hub -> the target's node (`fs.write`).
//!
//! Where the files land depends on the target root's kind (`layout_of`): `hf-home`/`hf-hub` get the Hugging Face
//! cache layout (`blobs/<sha256>`, relative `snapshots/<commit>/<path>` links and `refs/<revision>`, created only once
//! every blob is in place); `gguf`/`plain` the files themselves; `auto` whatever its last scan showed.
//!
//! The worker of a row runs as a task; a pause or cancel cancels it at its next await (as asyncio's CancelledError
//! does) and the row's state, the partial clean-up and the progress event follow, exactly as in Python.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use base64::Engine as _;
use bytes::Bytes;
use regex::Regex;
use rusqlite::types::Value as SqlValue;
use serde_json::{Map, Value};
use sha2::Digest as _;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::domain::errors::{
    AppError, AppResult, NodeError, Obj, Problem, bad_request, conflict, not_found, unprocessable,
};
use crate::domain::fleet::{CopyRequest, FleetPort, NodeClient, NodeStream, StreamItem};
use crate::domain::models::{Download, DownloadRequest, Page};
use crate::domain::ports::{ByteStream, HubClient, RepoFile};
use crate::domain::repo_id::{check_repo_id, require_repo_id};
use crate::services::common::{EventBus, fnmatchcase, new_id, now, scrub};
use crate::services::gguf_info::{py, shard_names};
use crate::services::hf::HfService;
use crate::services::library::{CopySource, HostFacts, LibraryService, RootSpec, RootUsage};
use crate::services::nodeerr::problem_from_node;
use crate::services::settings::SettingsService;
use crate::services::store::{SqlRow, Store, paginate};

pub const TOPIC: &str = "downloads";
pub const DISK_MARGIN: f64 = 1.02;
/// a node must accept a chunk, or answer a small call, within this long
pub const STALL_S: f64 = 120.0;
/// after the last byte the node hashes the whole file: allow this slow a disk
pub const VERIFY_BPS: f64 = 20e6;
/// paths per fs.stat (proto.ARGS)
pub const STAT_BATCH: usize = 256;
pub const MAX_PATTERNS: usize = 64;
/// root kinds an HF repository may be downloaded or copied into
pub const HF_KINDS: &[&str] = &["hf-home", "hf-hub", "gguf", "auto"];
pub const HF_LAYOUTS: &[&str] = &["hf-home", "hf-hub"];
/// Names a model directory usually holds (the node protocol has no directory listing).
pub const PLAIN_NAMES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "tokenizer.tiktoken",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "added_tokens.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "bpe.model",
    "chat_template.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
    "video_preprocessor_config.json",
    "feature_extractor_config.json",
    "quantization_config.json",
    "hf_quant_config.json",
    "adapter_config.json",
    "params.json",
    "tekken.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "consolidated.safetensors",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "NOTICE",
    "USE_POLICY.md",
    ".gitattributes",
];
pub const PLAIN_INDEXES: &[&str] = &[
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
];
pub const INDEX_MAX: usize = 4 * 1024 * 1024;
const MAX_DATA: usize = ec_proto::MAX_DATA;

static SHA256_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A[0-9a-f]{64}\z").unwrap_or_else(|_| unreachable!()));
static COMMIT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A[0-9a-f]{40}\z").unwrap_or_else(|_| unreachable!()));
/// An HF cache directory as a node lists it: [hub/]models--org--name, optionally down to one snapshot file.
static HF_REL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\A(?P<repo>(?:hub/)?models--(?P<org>[^/]+?)--(?P<name>[^/]+))(?:/snapshots/(?P<rev>[^/]+)(?:/(?P<path>.+))?)?\n?\z",
    )
    .unwrap_or_else(|_| unreachable!())
});
static NOT_DIRNAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^A-Za-z0-9._-]").unwrap_or_else(|_| unreachable!()));

/// Hub-supplied file names are untrusted: no absolute paths, '..', backslashes or NULs.
pub fn safe_rel_path(path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') || path.contains('\\') || path.contains('\0') {
        return false;
    }
    path.split('/').all(|seg| !matches!(seg, "" | "." | ".."))
}

/// `_match`: no patterns = everything; an exact path wins over fnmatch's reading of `[..]`.
pub fn path_matches(path: &str, patterns: Option<&[String]>) -> bool {
    match patterns {
        None | Some([]) => true,
        Some(p) => p.iter().any(|x| path == x || fnmatchcase(path, x)),
    }
}

// ---- layouts (pure, unit-tested) ---------------------------------------------------------------------------------
/// The repo's cache directory inside a root: under `hub/` for an HF_HOME, directly in a hub cache.
pub fn hf_repo_rel(layout: &str, repo_id: &str) -> String {
    let folder = format!("models--{}", repo_id.replace('/', "--"));
    if layout == "hf-home" {
        format!("hub/{folder}")
    } else {
        folder
    }
}

pub fn files_dir(repo_id: &str) -> String {
    repo_id.replace('/', "--")
}

/// LFS files are content addressed (their sha256). A small file has no hash from the hub, so its blob is named
/// after the commit and path: a newer commit's config.json is a new blob, never the stale one.
pub fn blob_name(f: &RepoFile, commit: &str) -> String {
    match &f.sha256 {
        Some(s) if !s.is_empty() => s.clone(),
        _ => {
            let d = sha2::Sha256::digest(format!("{commit}\0{}", f.path).as_bytes());
            format!("nolfs-{}", &hex::encode(d)[..40])
        }
    }
}

/// (link path, relative target) of one snapshot file, both inside the root.
pub fn snapshot_link(repo_rel: &str, commit: &str, path: &str, blob: &str) -> (String, String) {
    let link = format!("{repo_rel}/snapshots/{commit}/{path}");
    let target = py::relpath(&format!("{repo_rel}/blobs/{blob}"), &py::dirname(&link));
    (link, target)
}

/// How a transfer writes into a root ("hf-home" | "hf-hub" | "files"), or None when the root cannot hold one.
pub fn layout_of(spec: &RootSpec, usage: &RootUsage, empty_auto: &str) -> Option<&'static str> {
    match spec.kind.as_str() {
        "hf-home" => Some("hf-home"),
        "hf-hub" => Some("hf-hub"),
        "gguf" | "plain" => Some("files"),
        "auto" => {
            if usage.hf_home {
                Some("hf-home")
            } else if usage.hf_hub {
                Some("hf-hub")
            } else if usage.loose && !usage.ollama && !usage.other {
                Some("files")
            } else if usage.entries == 0 {
                Some(match empty_auto {
                    "files" => "files",
                    "hf-hub" => "hf-hub",
                    _ => "hf-home",
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// What a copy of a library copy is made of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyKind {
    /// a Hugging Face cache directory on its machine
    Hf {
        repo_id: String,
        repo_rel: String,
        revision: Option<String>,
        only: Option<Vec<String>>,
    },
    /// one .gguf file (or every shard of a split set) outside any HF cache
    Gguf { rels: Vec<String> },
    /// a plain model directory whose scan counted every file
    Plain {
        rel: String,
        name: String,
        files: i64,
        size: i64,
    },
}

/// `classify_copy_source`: None = not copyable with the node protocol as it is.
pub fn classify_copy_source(src: &CopySource) -> Option<CopyKind> {
    let rel = src.copy.rel.as_str();
    if let Some(m) = HF_REL.captures(rel) {
        let org = m.name("org").map_or("", |x| x.as_str());
        let name = m.name("name").map_or("", |x| x.as_str());
        let repo_id = format!("{org}/{name}");
        check_repo_id(&repo_id).ok()?;
        let commit = [
            m.name("rev").map(|x| x.as_str().to_string()),
            src.revision.clone(),
        ]
        .into_iter()
        .flatten()
        .find(|c| !c.is_empty() && COMMIT_RE.is_match(c));
        let mut only = None;
        if let Some(p) = m.name("path") {
            if !py::ends_with_ci(p.as_str(), ".gguf") {
                return None;
            }
            only = Some(shard_names(p.as_str(), src.shards));
        }
        return Some(CopyKind::Hf {
            repo_id,
            repo_rel: m.name("repo").map_or("", |x| x.as_str()).to_string(),
            revision: commit,
            only,
        });
    }
    if src.format == "gguf" && py::ends_with_ci(rel, ".gguf") && safe_rel_path(rel) {
        return Some(CopyKind::Gguf {
            rels: shard_names(rel, src.shards),
        });
    }
    if src.format == "plain" && (rel.is_empty() || safe_rel_path(rel)) {
        let counted = src.copy.files;
        let exact = counted.is_some_and(|c| c > 0)
            && !src.warnings.iter().any(|w| w.contains("lower bound"));
        // without exact totals nothing could prove a copy complete: not copyable
        return exact.then(|| CopyKind::Plain {
            rel: rel.to_string(),
            name: src.name.clone(),
            files: counted.unwrap_or(0),
            size: src.copy.size_bytes,
        });
    }
    None
}

// ---- plans ---------------------------------------------------------------------------------------------------------
#[derive(Clone)]
struct Target {
    host_id: String,
    label: String,
    root_id: String,
    root_path: String,
    layout: &'static str,
    kind: String,
    node: Arc<dyn NodeClient>,
}

#[derive(Debug, Clone)]
struct Item {
    path: String,
    size: i64,
    sha256: Option<String>,
    dest: String,
    link: Option<(String, String)>,
    from_hf: Option<(String, String, String)>,
    from_node: Option<(String, String, String)>,
}

#[derive(Debug, Clone)]
struct Plan {
    items: Vec<Item>,
    layout: &'static str,
    commit: Option<String>,
    repo_rel: Option<String>,
    refs_name: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct Stat {
    exists: bool,
    size: i64,
    partial: i64,
}

/// How a worker ended other than with success.
#[derive(Debug)]
enum RunErr {
    Node(NodeError),
    Problem(Problem),
    /// an unexpected failure: (Python type name, message)
    Other(&'static str, String),
}

impl From<AppError> for RunErr {
    fn from(e: AppError) -> Self {
        match e {
            AppError::Node(n) => Self::Node(n),
            AppError::Problem(p) => Self::Problem(p),
        }
    }
}

impl From<Problem> for RunErr {
    fn from(p: Problem) -> Self {
        Self::Problem(p)
    }
}

impl From<NodeError> for RunErr {
    fn from(e: NodeError) -> Self {
        Self::Node(e)
    }
}

type RunResult<T> = Result<T, RunErr>;

fn problem(status: u16, code: &str, detail: impl Into<String>) -> Problem {
    Problem::new(status, code, detail)
}

pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

struct Task {
    handle: JoinHandle<()>,
    token: CancellationToken,
}

struct Inner {
    db: Arc<Store>,
    hub: Arc<dyn HubClient>,
    hf: Arc<HfService>,
    bus: Arc<EventBus>,
    fleet: Arc<dyn FleetPort>,
    library: Arc<LibraryService>,
    sem: Arc<Semaphore>,
    tasks: Mutex<HashMap<String, Task>>,
    intent: Mutex<HashMap<String, String>>,
    max_queue: i64,
    clock: Mutex<Clock>,
}

pub struct DownloadService {
    inner: Arc<Inner>,
    settings: Arc<SettingsService>,
}

impl std::fmt::Debug for DownloadService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DownloadService")
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn monotonic() -> f64 {
    static START: LazyLock<std::time::Instant> = LazyLock::new(std::time::Instant::now);
    // time.monotonic() starts at an arbitrary large value; only differences matter
    1.0e6 + START.elapsed().as_secs_f64()
}

impl DownloadService {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        db: Arc<Store>,
        hub: Arc<dyn HubClient>,
        hf: Arc<HfService>,
        settings: Arc<SettingsService>,
        bus: Arc<EventBus>,
        fleet: Arc<dyn FleetPort>,
        library: Arc<LibraryService>,
        concurrency: i64,
        max_queue: i64,
    ) -> Self {
        let permits = usize::try_from(concurrency.max(1)).unwrap_or(1);
        Self {
            inner: Arc::new(Inner {
                db,
                hub,
                hf,
                bus,
                fleet,
                library,
                sem: Arc::new(Semaphore::new(permits)),
                tasks: Mutex::new(HashMap::new()),
                intent: Mutex::new(HashMap::new()),
                max_queue,
                clock: Mutex::new(Arc::new(monotonic)),
            }),
            settings,
        }
    }

    /// The clock progress is measured with (tests drive it; `time.monotonic` by default).
    pub fn set_clock(&self, clock: Clock) {
        *lock(&self.inner.clock) = clock;
    }

    /// Start-up: rows left running by a previous process become resumable (paused) again.
    pub fn recover(&self) -> AppResult<()> {
        let _ = &self.settings;
        self.inner.db.execute(
            "UPDATE downloads SET state='paused' WHERE state IN ('queued','running')",
            &[],
        )?;
        Ok(())
    }

    pub fn get(&self, did: &str) -> AppResult<Download> {
        self.inner.row(did)
    }

    pub fn list_page(&self, limit: i64, cursor: Option<&str>) -> AppResult<Page<Download>> {
        let (limit, off) = paginate(limit, cursor);
        let rows = self.inner.db.all(
            "SELECT * FROM downloads ORDER BY created_at DESC LIMIT ? OFFSET ?",
            &[&(limit + 1), &off],
        )?;
        let lim = usize::try_from(limit).unwrap_or(0);
        let more = rows.len() > lim;
        Ok(Page::new(
            rows.iter()
                .take(lim)
                .map(|r| to_model(r, 0.0, None))
                .collect(),
            more.then(|| (off + limit).to_string()),
        ))
    }

    pub async fn create(&self, req: &DownloadRequest) -> AppResult<Download> {
        let inner = &self.inner;
        require_repo_id(&req.repo_id)?;
        if let Some(pats) = req.allow_patterns.as_ref().filter(|p| !p.is_empty())
            && (pats.len() > MAX_PATTERNS
                || pats.iter().any(|p| p.is_empty() || p.chars().count() > 256))
        {
            return Err(bad_request(format!(
                "allow_patterns: at most {MAX_PATTERNS} patterns of 1-256 characters"
            ))
            .code("invalid_params")
            .into());
        }
        inner.check_capacity()?;
        let target = inner.resolve(
            &req.host_id,
            req.root_id.as_deref(),
            HF_KINDS,
            &["hf-home", "hf-hub", "files"],
            HF_LAYOUTS,
            "a Hugging Face repository",
            &[],
            "hf-home",
        )?;
        inner.check_duplicate("hf", &req.repo_id, &target.host_id, &target.root_id, None)?;
        let (did, ts) = (new_id("dl_"), now());
        let pats = req
            .allow_patterns
            .as_ref()
            .filter(|p| !p.is_empty())
            .map(|p| crate::pyjson::dumps(&Value::from(p.clone())));
        inner.db.execute(
            "INSERT INTO downloads(id,repo_id,revision,allow_patterns,state,created_at,updated_at,host_id,root_id,kind) \
             VALUES(?,?,?,?,?,?,?,?,?,?)",
            &[
                &did,
                &req.repo_id,
                &req.revision,
                &pats,
                &"queued",
                &ts,
                &ts,
                &target.host_id,
                &target.root_id,
                &"hf",
            ],
        )?;
        // Python returns the row before the new task has taken a step: read it first (no race with the worker)
        let row = inner.row(&did)?;
        Inner::start(inner, &did);
        Ok(row)
    }

    /// Copy a library copy to another machine (or another root of the same one): a row of kind "copy".
    pub async fn copy(&self, copy_id: &str, req: &CopyRequest) -> AppResult<Download> {
        let inner = &self.inner;
        let Some(src) = inner.library.source(copy_id) else {
            return Err(not_found("no such model copy")
                .code("copy_not_found")
                .into());
        };
        let Some(kind) = classify_copy_source(&src) else {
            return Err(problem(
                501,
                "copy_unsupported",
                format!(
                    "a {} model of this kind cannot be copied yet: only Hugging Face cache folders, .gguf files and \
                     plain model directories whose scan counted every file can be (the node protocol has no \
                     directory listing to enumerate anything else)",
                    src.format
                ),
            )
            .into());
        };
        inner.check_capacity()?;
        let src_node = inner.fleet.node(&src.copy.host_id)?;
        let state = src_node.link().state;
        if state != "up" {
            return Err(problem_from_node(
                &NodeError::unavailable(
                    &src.copy.host_id,
                    format!("it is not connected ({state})"),
                ),
                Some(&src_node.host().label),
            )
            .into());
        }
        let target =
            inner.resolve_copy_target(&req.host_id, req.root_id.as_deref(), &kind, &src)?;
        inner.check_duplicate(
            "copy",
            &src.name,
            &target.host_id,
            &target.root_id,
            Some(copy_id),
        )?;
        let (did, ts) = (new_id("dl_"), now());
        inner.db.execute(
            "INSERT INTO downloads(id,repo_id,revision,allow_patterns,state,created_at,updated_at,host_id,root_id,kind,\
             source_copy_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            &[
                &did,
                &src.name,
                &src.revision,
                &None::<String>,
                &"queued",
                &ts,
                &ts,
                &target.host_id,
                &target.root_id,
                &"copy",
                &copy_id,
            ],
        )?;
        // Python returns the row before the new task has taken a step: read it first (no race with the worker)
        let row = inner.row(&did)?;
        Inner::start(inner, &did);
        Ok(row)
    }

    pub async fn pause(&self, did: &str) -> AppResult<Download> {
        let inner = &self.inner;
        let dl = inner.row(did)?;
        if dl.state != "queued" && dl.state != "running" {
            return Err(conflict(format!("cannot pause a {} download", dl.state)).into());
        }
        inner.stop_task(did, "paused").await;
        let st = inner.row(did)?.state;
        if st == "queued" || st == "running" {
            // cancelled before its first step: nobody else will say so
            inner.set(did, &[("state", text("paused"))]);
        }
        inner.row(did)
    }

    pub async fn resume(&self, did: &str) -> AppResult<Download> {
        let inner = &self.inner;
        let dl = inner.row(did)?;
        if dl.state != "paused" && dl.state != "failed" {
            return Err(conflict(format!("cannot resume a {} download", dl.state)).into());
        }
        inner.set(
            did,
            &[
                ("state", text("queued")),
                ("error", SqlValue::Null),
                ("error_code", SqlValue::Null),
            ],
        );
        let row = inner.row(did)?;
        Inner::start(inner, did);
        Ok(row)
    }

    pub async fn cancel(&self, did: &str) -> AppResult<Download> {
        let inner = &self.inner;
        let dl = inner.row(did)?;
        if dl.state == "completed" || dl.state == "cancelled" {
            return Err(conflict(format!("download already {}", dl.state)).into());
        }
        let live = inner.is_live(did);
        inner.stop_task(did, "cancelled").await;
        inner.set(did, &[("state", text("cancelled"))]);
        if !live {
            // a live task cleans up in its own cancel handler
            inner.cleanup_partial(did).await;
        }
        inner.publish(did, 0.0, None);
        inner.row(did)
    }

    pub async fn delete(&self, did: &str) -> AppResult<()> {
        let inner = &self.inner;
        let dl = inner.row(did)?;
        let live = inner.is_live(did);
        inner.stop_task(did, "cancelled").await;
        if dl.state != "completed" && !live {
            inner.cleanup_partial(did).await;
        }
        inner
            .db
            .execute("DELETE FROM downloads WHERE id=?", &[&did])?;
        Ok(())
    }

    /// Test/ops helper: block until the download task ends.
    pub async fn wait(&self, did: &str) {
        loop {
            let done = lock(&self.inner.tasks)
                .get(did)
                .is_none_or(|t| t.handle.is_finished());
            if done {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    pub async fn shutdown(&self) {
        let ids: Vec<String> = lock(&self.inner.tasks).keys().cloned().collect();
        for did in ids {
            self.inner.stop_task(&did, "paused").await;
        }
    }
}

fn text(s: &str) -> SqlValue {
    SqlValue::Text(s.to_string())
}

fn opt_text(s: Option<&str>) -> SqlValue {
    s.map_or(SqlValue::Null, |x| SqlValue::Text(x.to_string()))
}

fn to_model(r: &SqlRow, speed: f64, eta: Option<f64>) -> Download {
    let pats = r
        .text("allow_patterns")
        .filter(|t| !t.is_empty())
        .and_then(|t| serde_json::from_str::<Vec<String>>(&t).ok());
    Download {
        id: r.str_or("id", ""),
        repo_id: r.str_or("repo_id", ""),
        revision: r.text("revision"),
        commit_sha: r.text("commit_sha"),
        allow_patterns: pats,
        state: r.str_or("state", ""),
        total_bytes: r.i64("total_bytes").unwrap_or(0),
        done_bytes: r.i64("done_bytes").unwrap_or(0),
        speed_bps: speed,
        eta_s: eta,
        files_total: r.i64("files_total").unwrap_or(0),
        files_done: r.i64("files_done").unwrap_or(0),
        current_file: r.text("current_file"),
        error: r.text("error"),
        error_code: r.text("error_code"),
        created_at: r.f64("created_at").unwrap_or(0.0),
        updated_at: r.f64("updated_at").unwrap_or(0.0),
        host_id: r.str_or("host_id", "local"),
        root_id: r.text("root_id"),
        kind: r.str_or("kind", "hf"),
        source_copy_id: r.text("source_copy_id"),
    }
}

/// File and byte progress of one running row: persisted about once a second, published about four times a second.
struct Progress {
    did: String,
    total: i64,
    done: i64,
    base: i64,
    started: f64,
    last_pub: f64,
    last_db: f64,
    speed: f64,
}

impl Inner {
    fn now_mono(&self) -> f64 {
        let c = Arc::clone(&lock(&self.clock));
        c()
    }

    fn row(&self, did: &str) -> AppResult<Download> {
        match self.db.one("SELECT * FROM downloads WHERE id=?", &[&did])? {
            Some(r) => Ok(to_model(&r, 0.0, None)),
            None => Err(not_found(format!("no such download: {did}")).into()),
        }
    }

    /// `_set(did, **cols)`: column names are literals of this module; updated_at always moves.
    fn set(&self, did: &str, cols: &[(&str, SqlValue)]) {
        let mut sets: Vec<String> = cols.iter().map(|(k, _)| format!("{k}=?")).collect();
        sets.push("updated_at=?".into());
        let mut vals: Vec<SqlValue> = cols.iter().map(|(_, v)| v.clone()).collect();
        vals.push(SqlValue::Real(now()));
        vals.push(text(did));
        let refs: Vec<&dyn rusqlite::ToSql> =
            vals.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
        let _ = self.db.execute(
            &format!("UPDATE downloads SET {} WHERE id=?", sets.join(", ")),
            &refs,
        );
    }

    fn publish(&self, did: &str, speed: f64, eta: Option<f64>) {
        if let Ok(Some(r)) = self.db.one("SELECT * FROM downloads WHERE id=?", &[&did]) {
            self.bus.publish(
                TOPIC,
                serde_json::to_value(to_model(&r, speed, eta)).unwrap_or(Value::Null),
            );
        }
    }

    fn label(&self, host_id: &str) -> String {
        self.library.host_label(host_id)
    }

    /// The machine and root a transfer writes into, or the problem that says why not.
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        &self,
        host_id: &str,
        root_id: Option<&str>,
        kinds: &[&str],
        accept: &[&str],
        default: &[&str],
        what: &str,
        exclude: &[(String, String)],
        empty_auto: &str,
    ) -> AppResult<Target> {
        let node = self.fleet.node(host_id)?;
        let label = node.host().label.clone();
        let state = node.link().state;
        if state != "up" {
            return Err(problem_from_node(
                &NodeError::unavailable(host_id, format!("it is not connected ({state})")),
                Some(&label),
            )
            .into());
        }
        let Some(facts): Option<HostFacts> = self.library.host_facts(host_id) else {
            return Err(problem_from_node(
                &NodeError::unavailable(host_id, "its node has not reported yet"),
                Some(&label),
            )
            .into());
        };
        if facts.mode != "manage" {
            return Err(problem(
                409,
                "monitor_only",
                format!("{label}: this machine's node policy is monitor-only, so nothing can be written to it"),
            )
            .with("host_id", host_id)
            .into());
        }
        let excluded = |id: &str| exclude.iter().any(|(h, r)| h == host_id && r == id);
        let chosen: (RootSpec, &'static str) = if let Some(root_id) = root_id {
            let Some(spec) = facts.root(root_id).cloned() else {
                return Err(unprocessable(format!(
                    "{label} has no model root {}",
                    ec_proto::pyfmt::repr_str(&py::take(root_id, 40))
                ))
                .code("unknown_root")
                .into());
            };
            let rid = ec_proto::pyfmt::repr_str(&spec.id);
            if excluded(&spec.id) {
                return Err(conflict(format!(
                    "{label}: model root {rid} is where the model already is (or storage it shares)"
                ))
                .code("copy_same_location")
                .into());
            }
            if !spec.writable {
                return Err(conflict(format!(
                    "{label}: model root {rid} is read-only in its node policy"
                ))
                .code("root_not_writable")
                .into());
            }
            let layout = layout_of(
                &spec,
                &self.library.root_usage(host_id, &spec.id),
                empty_auto,
            );
            match layout {
                Some(l) if kinds.contains(&spec.kind.as_str()) && accept.contains(&l) => (spec, l),
                _ => {
                    return Err(unprocessable(format!(
                        "{label}: model root {rid} (kind {}) cannot hold {what}; use a root of kind {}",
                        spec.kind,
                        kinds.join(", ")
                    ))
                    .code("root_kind_unsupported")
                    .into());
                }
            }
        } else {
            let mut found = None;
            for spec in &facts.roots {
                let layout = if spec.writable && kinds.contains(&spec.kind.as_str()) {
                    layout_of(
                        spec,
                        &self.library.root_usage(host_id, &spec.id),
                        empty_auto,
                    )
                } else {
                    None
                };
                if let Some(l) = layout
                    && default.contains(&l)
                    && !excluded(&spec.id)
                {
                    found = Some((spec.clone(), l));
                    break;
                }
            }
            match found {
                Some(c) => c,
                None => {
                    return Err(conflict(format!(
                        "{label} has no writable model root that can hold {what}"
                    ))
                    .code("no_writable_root")
                    .into());
                }
            }
        };
        let (spec, layout) = chosen;
        let _ = facts.allow_delete;
        Ok(Target {
            host_id: host_id.to_string(),
            label,
            root_id: spec.id.clone(),
            root_path: spec.path.clone(),
            layout,
            kind: spec.kind,
            node,
        })
    }

    /// A repository's cache copies into an HF cache (or, flattened, into a gguf root); a loose .gguf file only into
    /// a gguf root; a plain model directory only into a plain root. Never onto the storage the copy is already on.
    fn resolve_copy_target(
        &self,
        host_id: &str,
        root_id: Option<&str>,
        kind: &CopyKind,
        src: &CopySource,
    ) -> AppResult<Target> {
        match kind {
            CopyKind::Hf { .. } => self.resolve(
                host_id,
                root_id,
                HF_KINDS,
                &["hf-home", "hf-hub", "files"],
                HF_LAYOUTS,
                "that model",
                &src.locations,
                "hf-home",
            ),
            CopyKind::Gguf { .. } => self.resolve(
                host_id,
                root_id,
                &["gguf", "auto"],
                &["files"],
                &["files"],
                "a loose .gguf file",
                &src.locations,
                "files",
            ),
            CopyKind::Plain { .. } => self.resolve(
                host_id,
                root_id,
                &["plain", "auto"],
                &["files"],
                &["files"],
                "a plain model directory",
                &src.locations,
                "files",
            ),
        }
    }

    fn check_capacity(&self) -> AppResult<()> {
        let n = self
            .db
            .one(
                "SELECT COUNT(*) AS n FROM downloads WHERE state IN ('queued','running','paused')",
                &[],
            )?
            .and_then(|r| r.i64("n"))
            .unwrap_or(0);
        if n >= self.max_queue {
            return Err(
                conflict("too many unfinished downloads; cancel or finish some first")
                    .code("download_queue_full")
                    .into(),
            );
        }
        Ok(())
    }

    /// Two rows writing the same files into the same root would fight over one `.incomplete` partial.
    fn check_duplicate(
        &self,
        kind: &str,
        name: &str,
        host_id: &str,
        root_id: &str,
        source: Option<&str>,
    ) -> AppResult<()> {
        let row = self.db.one(
            "SELECT 1 FROM downloads WHERE kind=? AND repo_id=? AND host_id=? AND root_id=? AND \
             COALESCE(source_copy_id,'')=? AND state IN ('queued','running','paused')",
            &[&kind, &name, &host_id, &root_id, &source.unwrap_or("")],
        )?;
        if row.is_some() {
            return Err(conflict(
                "this is already being downloaded to that location (queued, running or paused)",
            )
            .code("download_exists")
            .into());
        }
        Ok(())
    }

    fn start(me: &Arc<Self>, did: &str) {
        lock(&me.intent).remove(did);
        let token = CancellationToken::new();
        let (me2, d2, t2) = (Arc::clone(me), did.to_string(), token.clone());
        let handle = tokio::spawn(async move { Self::run(me2, d2, t2).await });
        if let Some(old) = lock(&me.tasks).insert(did.to_string(), Task { handle, token }) {
            old.token.cancel();
        }
    }

    /// A task that has not ended (a failed or finished row keeps its dead task until it is stopped).
    fn is_live(&self, did: &str) -> bool {
        lock(&self.tasks)
            .get(did)
            .is_some_and(|t| !t.handle.is_finished())
    }

    async fn stop_task(&self, did: &str, intent: &str) {
        let task = lock(&self.tasks).remove(did);
        lock(&self.intent).insert(did.to_string(), intent.to_string());
        match task {
            Some(t) if !t.handle.is_finished() => {
                t.token.cancel();
                let _ = t.handle.await;
            }
            _ => {
                if intent == "paused" {
                    self.set(did, &[("state", text("paused"))]);
                }
            }
        }
    }

    // -- worker ------------------------------------------------------------------------------------------------
    /// (error_code, text) of a node refusal. A full disk reads the same whoever noticed it first.
    fn failure(&self, e: &NodeError) -> (String, String) {
        let code = if e.code == "no_space" {
            "insufficient_disk".to_string()
        } else {
            e.code.clone()
        };
        (
            code,
            py::take(
                &scrub(&format!("{}: {}", self.label(&e.host_id), e.message)),
                500,
            ),
        )
    }

    async fn run(me: Arc<Self>, did: String, token: CancellationToken) {
        let Ok(dl) = me.row(&did) else {
            return;
        };
        let work = async {
            let _permit = me
                .sem
                .acquire()
                .await
                .map_err(|_| RunErr::Other("RuntimeError", "closed".into()))?;
            me.set(&did, &[("state", text("running"))]);
            if dl.kind == "copy" {
                me.copy_run(&did, &dl).await
            } else {
                me.download_run(&did, &dl).await
            }
        };
        // biased, cancellation first: a row cancelled before its task took a step never takes one (asyncio
        // delivers CancelledError at the first await)
        // a panic in the run is Python's `except Exception` around `_run`: the row fails, the worker never dies silently
        let work = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(work));
        let res = tokio::select! {
            biased;
            () = token.cancelled() => None,
            r = work => Some(r.unwrap_or_else(|p| Err(RunErr::Other(
                "panic",
                crate::panics::redacted(&crate::panics::payload_text(&*p)),
            )))),
        };
        match res {
            None => {
                let intent = lock(&me.intent)
                    .get(&did)
                    .cloned()
                    .unwrap_or_else(|| "paused".into());
                me.set(&did, &[("state", text(&intent))]);
                if intent == "cancelled" {
                    me.cleanup_partial(&did).await;
                }
                me.publish(&did, 0.0, None);
            }
            Some(Ok(())) => {}
            Some(Err(RunErr::Node(e))) => {
                let (code, txt) = me.failure(&e);
                me.set(
                    &did,
                    &[
                        ("state", text("failed")),
                        ("error", text(&txt)),
                        ("error_code", text(&code)),
                    ],
                );
                me.publish(&did, 0.0, None);
            }
            Some(Err(RunErr::Problem(p))) => {
                me.set(
                    &did,
                    &[
                        ("state", text("failed")),
                        ("error", text(&py::take(&scrub(&p.detail), 500))),
                        ("error_code", text(&p.code)),
                    ],
                );
                me.publish(&did, 0.0, None);
            }
            Some(Err(RunErr::Other(name, msg))) => {
                tracing::error!(target: "engine_console.services.downloads", "download {} crashed", did);
                me.set(
                    &did,
                    &[
                        ("state", text("failed")),
                        (
                            "error",
                            text(&py::take(&format!("{name}: {}", scrub(&msg)), 500)),
                        ),
                        ("error_code", text("download_failed")),
                    ],
                );
                me.publish(&did, 0.0, None);
            }
        }
    }

    /// Remove the file in flight's `.incomplete` partial, when the node's policy allows deletes (best effort).
    async fn cleanup_partial(&self, did: &str) {
        let Ok(Some(r)) = self.db.one(
            "SELECT host_id, root_id, current_dest FROM downloads WHERE id=?",
            &[&did],
        ) else {
            return;
        };
        let (Some(host), Some(root), Some(dest)) = (
            r.text("host_id"),
            r.text("root_id").filter(|x| !x.is_empty()),
            r.text("current_dest").filter(|x| !x.is_empty()),
        ) else {
            return;
        };
        let Some(facts) = self.library.host_facts(&host) else {
            return;
        };
        if !facts.allow_delete || facts.mode != "manage" {
            return;
        }
        let Ok(node) = self.fleet.node(&host) else {
            return;
        };
        let mut args = Obj::new();
        args.insert("root".into(), Value::String(root));
        args.insert("rel".into(), Value::String(format!("{dest}.incomplete")));
        let call = node.call("fs.delete", Some(args), Duration::from_secs_f64(STALL_S));
        if let Ok(Ok(_)) = tokio::time::timeout(Duration::from_secs_f64(STALL_S + 10.0), call).await
        {
            self.set(did, &[("current_dest", SqlValue::Null)]);
        }
    }

    // -- planning ------------------------------------------------------------------------------------------------
    /// The commit the listing is for, after the hub-supplied names, sizes and hashes have been vetted.
    fn checked_listing(files: &[RepoFile], sha: Option<&str>) -> RunResult<String> {
        let Some(sha) = sha.filter(|s| !s.is_empty() && COMMIT_RE.is_match(s)) else {
            return Err(problem(
                500,
                "hf_bad_sha",
                "hub returned a missing or malformed commit sha",
            )
            .into());
        };
        for f in files {
            if !safe_rel_path(&f.path) {
                return Err(problem(
                    500,
                    "hf_bad_path",
                    "hub listed an unsafe file path; refusing the whole download",
                )
                .into());
            }
            if let Some(h) = &f.sha256
                && !SHA256_RE.is_match(h)
            {
                return Err(problem(
                    500,
                    "hf_bad_hash",
                    format!(
                        "hub returned a malformed sha256 for {}",
                        py::take(&f.path, 80)
                    ),
                )
                .into());
            }
            if f.size < 0 {
                return Err(
                    problem(500, "hf_bad_size", "hub returned a negative file size").into(),
                );
            }
        }
        Ok(sha.to_string())
    }

    /// The plan for `files` of one repo at one commit; `source(f)` says where each file's bytes come from, or None
    /// to leave a file out.
    fn hf_items(
        target: &Target,
        repo_id: &str,
        commit: &str,
        files: &[RepoFile],
        source: impl Fn(&RepoFile) -> Option<(bool, (String, String, String))>,
    ) -> Plan {
        let repo_rel = hf_repo_rel(target.layout, repo_id);
        let mut items = Vec::new();
        for f in files {
            let Some((from_hf, coords)) = source(f) else {
                continue;
            };
            let mut item = Item {
                path: f.path.clone(),
                size: f.size,
                sha256: f.sha256.clone(),
                dest: String::new(),
                link: None,
                from_hf: None,
                from_node: None,
            };
            if target.layout == "files" {
                item.dest = format!("{}/{}", files_dir(repo_id), f.path);
            } else {
                let blob = blob_name(f, commit);
                item.dest = format!("{repo_rel}/blobs/{blob}");
                item.link = Some(snapshot_link(&repo_rel, commit, &f.path, &blob));
            }
            if from_hf {
                item.from_hf = Some(coords);
            } else {
                item.from_node = Some(coords);
            }
            items.push(item);
        }
        Plan {
            items,
            layout: target.layout,
            commit: Some(commit.to_string()),
            repo_rel: Some(repo_rel),
            refs_name: "main".into(),
        }
    }

    async fn download_run(&self, did: &str, dl: &Download) -> RunResult<()> {
        let target = self.resolve(
            &dl.host_id,
            dl.root_id.as_deref(),
            HF_KINDS,
            &["hf-home", "hf-hub", "files"],
            HF_LAYOUTS,
            "a Hugging Face repository",
            &[],
            "hf-home",
        )?;
        let hub = self
            .hf
            .list_files(&dl.repo_id, dl.revision.as_deref())
            .await?;
        let commit = Self::checked_listing(&hub.files, hub.sha.as_deref())?;
        let files: Vec<RepoFile> = hub
            .files
            .iter()
            .filter(|f| path_matches(&f.path, dl.allow_patterns.as_deref()))
            .cloned()
            .collect();
        if files.is_empty() {
            return Err(bad_request("no files match allow_patterns")
                .code("no_files")
                .into());
        }
        if target.kind == "gguf" && !files.iter().any(|f| py::ends_with_ci(&f.path, ".gguf")) {
            return Err(bad_request(
                "a gguf model root holds .gguf files, and none of the selected files is one",
            )
            .code("no_gguf_files")
            .into());
        }
        let repo = dl.repo_id.clone();
        let c2 = commit.clone();
        let mut plan = Self::hf_items(&target, &dl.repo_id, &commit, &files, move |f| {
            Some((true, (repo.clone(), c2.clone(), f.path.clone())))
        });
        plan.refs_name = dl
            .revision
            .clone()
            .filter(|r| !r.is_empty())
            .unwrap_or_else(|| "main".into());
        self.execute(did, &target, plan).await
    }

    async fn copy_run(&self, did: &str, dl: &Download) -> RunResult<()> {
        let Some(src) = self
            .library
            .source(dl.source_copy_id.as_deref().unwrap_or(""))
        else {
            return Err(not_found(
                "the model copy this row was copying is no longer in the library",
            )
            .code("copy_source_gone")
            .into());
        };
        let Some(kind) = classify_copy_source(&src) else {
            return Err(problem(
                501,
                "copy_unsupported",
                "this model can no longer be copied",
            )
            .into());
        };
        let target = self.resolve_copy_target(&dl.host_id, dl.root_id.as_deref(), &kind, &src)?;
        let src_node = self.fleet.node(&src.copy.host_id)?;
        let state = src_node.link().state;
        if state != "up" {
            return Err(problem_from_node(
                &NodeError::unavailable(
                    &src.copy.host_id,
                    format!("it is not connected ({state})"),
                ),
                Some(&src_node.host().label),
            )
            .into());
        }
        let plan = match &kind {
            CopyKind::Hf { .. } => {
                self.plan_hf_copy(&target, &src, &kind, src_node.as_ref())
                    .await?
            }
            CopyKind::Gguf { rels } => self.plan_gguf_copy(&src, rels, src_node.as_ref()).await?,
            CopyKind::Plain {
                rel,
                name,
                files,
                size,
            } => {
                self.plan_plain_copy(&src, rel, name, *files, *size, src_node.as_ref())
                    .await?
            }
        };
        self.execute(did, &target, plan).await
    }

    /// Large files the source holds are copied machine to machine; the few small files are fetched from Hugging
    /// Face. A file the source lacks is skipped: a partial copy stays partial.
    async fn plan_hf_copy(
        &self,
        target: &Target,
        src: &CopySource,
        kind: &CopyKind,
        src_node: &dyn NodeClient,
    ) -> RunResult<Plan> {
        let CopyKind::Hf {
            repo_id,
            repo_rel,
            revision,
            only,
        } = kind
        else {
            return Err(RunErr::Other("AssertionError", String::new()));
        };
        let hub = self.hf.list_files(repo_id, revision.as_deref()).await?;
        let commit = Self::checked_listing(&hub.files, hub.sha.as_deref())?;
        let files: Vec<RepoFile> = hub
            .files
            .iter()
            .filter(|f| only.as_ref().is_none_or(|o| o.contains(&f.path)))
            .cloned()
            .collect();
        if files.is_empty() {
            return Err(
                bad_request("none of the model's files are in the repository listing")
                    .code("no_files")
                    .into(),
            );
        }
        let large: Vec<&RepoFile> = files
            .iter()
            .filter(|f| f.sha256.as_deref().is_some_and(|s| !s.is_empty()))
            .collect();
        let blob =
            |f: &RepoFile| format!("{repo_rel}/blobs/{}", f.sha256.clone().unwrap_or_default());
        let stats = self
            .stat(
                src_node,
                &src.copy.root_id,
                &large.iter().map(|f| blob(f)).collect::<Vec<_>>(),
            )
            .await?;
        let held: HashSet<String> = large
            .iter()
            .filter(|f| {
                let st = stats.get(&blob(f)).copied().unwrap_or_default();
                st.exists && st.size == f.size
            })
            .map(|f| f.path.clone())
            .collect();
        if !large.is_empty() && held.is_empty() {
            return Err(problem(
                409,
                "copy_source_mismatch",
                "none of the repository's large files are on the source copy at the revision the hub lists (the copy \
                 is at another revision, or is incomplete)",
            )
            .into());
        }
        let (host, root, rr, rid, c2) = (
            src.copy.host_id.clone(),
            src.copy.root_id.clone(),
            repo_rel.clone(),
            repo_id.clone(),
            commit.clone(),
        );
        Ok(Self::hf_items(
            target,
            repo_id,
            &commit,
            &files,
            move |f| match f.sha256.as_deref().filter(|s| !s.is_empty()) {
                None => Some((true, (rid.clone(), c2.clone(), f.path.clone()))),
                Some(sha) if held.contains(&f.path) => Some((
                    false,
                    (host.clone(), root.clone(), format!("{rr}/blobs/{sha}")),
                )),
                Some(_) => None,
            },
        ))
    }

    async fn plan_gguf_copy(
        &self,
        src: &CopySource,
        rels: &[String],
        src_node: &dyn NodeClient,
    ) -> RunResult<Plan> {
        let stats = self.stat(src_node, &src.copy.root_id, rels).await?;
        let mut items = Vec::new();
        for rel in rels {
            let st = stats.get(rel).copied().unwrap_or_default();
            if !st.exists {
                let last = rel.rsplit('/').next().unwrap_or(rel);
                return Err(problem(
                    409,
                    "copy_source_mismatch",
                    format!(
                        "{} is missing on the source machine (a split set needs every shard)",
                        py::take(last, 120)
                    ),
                )
                .into());
            }
            items.push(Item {
                path: py::basename(rel).to_string(),
                size: st.size,
                sha256: None,
                dest: rel.clone(),
                link: None,
                from_hf: None,
                from_node: Some((
                    src.copy.host_id.clone(),
                    src.copy.root_id.clone(),
                    rel.clone(),
                )),
            });
        }
        Ok(Plan {
            items,
            layout: "files",
            commit: None,
            repo_rel: None,
            refs_name: "main".into(),
        })
    }

    /// A plain model directory cannot be listed, so its files are found by name and the enumeration is then PROVED
    /// complete against the byte total and file count the node's scan reported.
    async fn plan_plain_copy(
        &self,
        src: &CopySource,
        rel: &str,
        name: &str,
        files: i64,
        size: i64,
        src_node: &dyn NodeClient,
    ) -> RunResult<Plan> {
        let (host, root) = (src.copy.host_id.clone(), src.copy.root_id.clone());
        let base = if rel.is_empty() {
            String::new()
        } else {
            format!("{rel}/")
        };
        let names: Vec<String> = PLAIN_NAMES.iter().map(|s| (*s).to_string()).collect();
        let mut found: Vec<(String, Stat)> = self.probe(src_node, &root, &base, &names).await?;
        let mut shards: Vec<String> = Vec::new();
        for index in PLAIN_INDEXES {
            if let Some((_, st)) = found.iter().find(|(n, _)| n == index) {
                let st = *st;
                shards.extend(
                    self.weight_files(src_node, &root, &format!("{base}{index}"), st.size)
                        .await?,
                );
            }
        }
        let mut rest: Vec<String> = Vec::new();
        for s in shards {
            if !rest.contains(&s) && !found.iter().any(|(n, _)| *n == s) {
                rest.push(s);
            }
        }
        for (n, st) in self.probe(src_node, &root, &base, &rest).await? {
            match found.iter_mut().find(|(x, _)| *x == n) {
                Some(slot) => slot.1 = st,
                None => found.push((n, st)),
            }
        }
        let total: i64 = found.iter().map(|(_, st)| st.size).sum();
        let count = i64::try_from(found.len()).unwrap_or(i64::MAX);
        if count != files || total != size {
            return Err(problem(
                501,
                "copy_listing_incomplete",
                format!(
                    "found {count} of the directory's {files} files ({total} of {size} bytes) by their usual names; \
                     the node protocol cannot list a directory, so a copy would be incomplete (a file with an unusual \
                     name, a subfolder, or a change since the last scan: rescan, or copy it another way)"
                ),
            )
            .into());
        }
        let raw = py::basename(rel);
        let raw = if raw.is_empty() { name } else { raw };
        let cleaned = NOT_DIRNAME.replace_all(raw, "_");
        let mut dirname = py::take(cleaned.trim_start_matches('.'), 100);
        if dirname.is_empty() {
            dirname = "model".into();
        }
        found.sort_by(|a, b| a.0.cmp(&b.0));
        let items = found
            .into_iter()
            .map(|(n, st)| Item {
                path: n.clone(),
                size: st.size,
                sha256: None,
                dest: format!("{dirname}/{n}"),
                link: None,
                from_hf: None,
                from_node: Some((host.clone(), root.clone(), format!("{base}{n}"))),
            })
            .collect();
        Ok(Plan {
            items,
            layout: "files",
            commit: None,
            repo_rel: None,
            refs_name: "main".into(),
        })
    }

    /// The regular files among `names` (relative to `base`) that exist, with their sizes (in `names` order).
    async fn probe(
        &self,
        node: &dyn NodeClient,
        root: &str,
        base: &str,
        names: &[String],
    ) -> RunResult<Vec<(String, Stat)>> {
        let rels: Vec<String> = names.iter().map(|n| format!("{base}{n}")).collect();
        let stats = self.stat(node, root, &rels).await?;
        Ok(names
            .iter()
            .filter_map(|n| {
                let st = stats
                    .get(&format!("{base}{n}"))
                    .copied()
                    .unwrap_or_default();
                st.exists.then(|| (n.clone(), st))
            })
            .collect())
    }

    /// The shard file names a weight index lists. Untrusted JSON: only plain relative names survive.
    async fn weight_files(
        &self,
        node: &dyn NodeClient,
        root: &str,
        rel: &str,
        size: i64,
    ) -> RunResult<Vec<String>> {
        let too_big = || {
            RunErr::Problem(problem(
                501,
                "copy_listing_incomplete",
                "a weight index is larger than 4 MiB; its shards cannot be listed",
            ))
        };
        if usize::try_from(size).unwrap_or(usize::MAX) > INDEX_MAX {
            return Err(too_big());
        }
        let mut buf: Vec<u8> = Vec::new();
        let mut chunks = NodeChunks::open(node, root, rel, 0);
        while let Some(c) = chunks.next().await? {
            buf.extend_from_slice(&c);
            if buf.len() > INDEX_MAX {
                chunks.cancel().await;
                return Err(too_big());
            }
        }
        let Ok(doc) = serde_json::from_slice::<Value>(&buf) else {
            return Ok(Vec::new());
        };
        let names: BTreeSet<String> = match doc.get("weight_map") {
            Some(Value::Object(m)) => m
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            _ => BTreeSet::new(),
        };
        Ok(names
            .into_iter()
            .filter(|n| safe_rel_path(n) && n.chars().count() <= 200)
            .collect())
    }

    // -- node calls ----------------------------------------------------------------------------------------------
    async fn call(
        &self,
        node: &dyn NodeClient,
        op: &str,
        args: Obj,
        timeout: f64,
    ) -> RunResult<Obj> {
        match tokio::time::timeout(
            Duration::from_secs_f64(timeout + 10.0),
            node.call(op, Some(args), Duration::from_secs_f64(timeout)),
        )
        .await
        {
            Ok(r) => Ok(r?),
            Err(_) => Err(RunErr::Other("TimeoutError", String::new())),
        }
    }

    /// exists / size / partial of each path, `STAT_BATCH` at a time. A path the node does not mention does not exist.
    async fn stat(
        &self,
        node: &dyn NodeClient,
        root: &str,
        rels: &[String],
    ) -> RunResult<HashMap<String, Stat>> {
        let mut out = HashMap::new();
        for batch in rels.chunks(STAT_BATCH) {
            let mut args = Obj::new();
            args.insert("root".into(), Value::String(root.to_string()));
            args.insert(
                "paths".into(),
                Value::Array(batch.iter().map(|r| Value::String(r.clone())).collect()),
            );
            let res = self.call(node, "fs.stat", args, 30.0).await?;
            let mut by_rel: HashMap<String, Map<String, Value>> = HashMap::new();
            if let Some(Value::Array(items)) = res.get("items") {
                for it in items {
                    if let Value::Object(o) = it
                        && let Some(r) = o.get("rel").and_then(Value::as_str)
                    {
                        by_rel.insert(r.to_string(), o.clone());
                    }
                }
            }
            for rel in batch {
                let it = by_rel.get(rel).cloned().unwrap_or_default();
                let pos = |k: &str| -> i64 {
                    match it.get(k) {
                        Some(Value::Bool(true)) => 1,
                        Some(v) => py::as_int(v).filter(|x| *x > 0).unwrap_or(0),
                        None => 0,
                    }
                };
                out.insert(
                    rel.clone(),
                    Stat {
                        exists: it.get("exists") == Some(&Value::Bool(true))
                            && it.get("is_dir") != Some(&Value::Bool(true)),
                        size: pos("size"),
                        partial: pos("partial"),
                    },
                );
            }
        }
        Ok(out)
    }

    // -- execution -------------------------------------------------------------------------------------------------
    fn done_paths(&self, did: &str) -> BTreeSet<String> {
        let raw = self
            .db
            .one("SELECT done_json FROM downloads WHERE id=?", &[&did])
            .ok()
            .flatten()
            .and_then(|r| r.text("done_json"))
            .filter(|t| !t.is_empty())
            .and_then(|t| serde_json::from_str::<Value>(&t).ok());
        match raw {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
            _ => BTreeSet::new(),
        }
    }

    fn progress_publish(&self, p: &Progress, eta: Option<f64>) {
        if let Ok(d) = self.row(&p.did) {
            let mut v = serde_json::to_value(d).unwrap_or(Value::Null);
            if let Value::Object(o) = &mut v {
                o.insert("done_bytes".into(), Value::from(p.done));
                o.insert("speed_bps".into(), Value::from(p.speed));
                o.insert("eta_s".into(), eta.map_or(Value::Null, Value::from));
            }
            self.bus.publish(TOPIC, v);
        }
    }

    fn advance(&self, p: &mut Progress, n: i64) {
        p.done += n;
        let t = self.now_mono();
        if t - p.last_pub < 0.25 {
            return;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            p.speed = (p.done - p.base) as f64 / (t - p.started).max(1e-6);
        }
        #[allow(clippy::cast_precision_loss)]
        let eta = (p.speed > 0.0).then(|| (p.total - p.done) as f64 / p.speed);
        p.last_pub = t;
        if t - p.last_db >= 1.0 {
            self.set(&p.did, &[("done_bytes", SqlValue::Integer(p.done))]);
            p.last_db = t;
        }
        self.progress_publish(p, eta);
    }

    async fn execute(&self, did: &str, target: &Target, plan: Plan) -> RunResult<()> {
        let items = &plan.items;
        let total: i64 = items.iter().map(|i| i.size).sum();
        let dests: Vec<String> = items.iter().map(|i| i.dest.clone()).collect();
        let stats = self
            .stat(target.node.as_ref(), &target.root_id, &dests)
            .await?;
        let mut finished = self.done_paths(did);
        let mut complete: Vec<bool> = Vec::new();
        let (mut already, mut partial_sum) = (0i64, 0i64);
        for it in items {
            let st = stats.get(&it.dest).copied().unwrap_or_default();
            // A content-addressed blob that exists is that content. A file under its final name could be an older
            // file of the same size, so only one this row itself finished counts.
            let ok = st.exists
                && st.size == it.size
                && (plan.layout != "files" || finished.contains(&it.dest));
            complete.push(ok);
            if ok {
                already += it.size;
            } else if 0 < st.partial && st.partial <= it.size {
                partial_sum += st.partial;
            }
        }
        let need = total - already - partial_sum;
        if let Some(free) = self.library.free_bytes(&target.host_id, &target.root_path) {
            #[allow(clippy::cast_precision_loss)]
            if need as f64 * DISK_MARGIN > free as f64 {
                return Err(problem(
                    507,
                    "insufficient_disk",
                    format!(
                        "{}: need {:.1} GB but only {:.1} GB free at {}",
                        target.label,
                        need as f64 / 1e9,
                        free as f64 / 1e9,
                        target.root_path
                    ),
                )
                .into());
            }
        }
        let mut files_done = i64::try_from(complete.iter().filter(|c| **c).count()).unwrap_or(0);
        self.set(
            did,
            &[
                ("commit_sha", opt_text(plan.commit.as_deref())),
                ("total_bytes", SqlValue::Integer(total)),
                (
                    "files_total",
                    SqlValue::Integer(i64::try_from(items.len()).unwrap_or(0)),
                ),
                ("done_bytes", SqlValue::Integer(already + partial_sum)),
                ("files_done", SqlValue::Integer(files_done)),
            ],
        );
        self.publish(did, 0.0, None);
        let started = self.now_mono();
        let mut progress = Progress {
            did: did.to_string(),
            total,
            done: already + partial_sum,
            base: already + partial_sum,
            started,
            last_pub: 0.0,
            last_db: 0.0,
            speed: 0.0,
        };
        let mut written: HashSet<String> = HashSet::new();
        for (it, ok) in items.iter().zip(complete.iter()) {
            if *ok {
                continue;
            }
            if written.contains(&it.dest) {
                // two files with the same content share one blob
                progress.done += it.size;
                files_done += 1;
                continue;
            }
            let st = stats.get(&it.dest).copied().unwrap_or_default();
            self.set(
                did,
                &[
                    ("current_file", text(&it.path)),
                    ("current_dest", text(&it.dest)),
                ],
            );
            let partial = if 0 < st.partial && st.partial <= it.size {
                st.partial
            } else {
                0
            };
            self.transfer(target, it, partial, &mut progress).await?;
            written.insert(it.dest.clone());
            finished.insert(it.dest.clone());
            let done_json =
                crate::pyjson::dumps(&Value::from(finished.iter().cloned().collect::<Vec<_>>()));
            self.set(
                did,
                &[
                    ("done_json", text(&done_json)),
                    ("current_dest", SqlValue::Null),
                ],
            );
            files_done += 1;
            self.set(
                did,
                &[
                    ("files_done", SqlValue::Integer(files_done)),
                    ("done_bytes", SqlValue::Integer(progress.done)),
                ],
            );
        }
        if plan.layout != "files"
            && let (Some(repo_rel), Some(commit)) = (&plan.repo_rel, &plan.commit)
        {
            self.set(
                did,
                &[
                    ("current_file", text("snapshot links")),
                    ("current_dest", SqlValue::Null),
                ],
            );
            for it in items {
                self.link(target, it).await?;
            }
            let mut args = Obj::new();
            args.insert("root".into(), Value::String(target.root_id.clone()));
            args.insert(
                "rel".into(),
                Value::String(format!("{repo_rel}/refs/{}", plan.refs_name)),
            );
            args.insert(
                "content_b64".into(),
                Value::String(base64::engine::general_purpose::STANDARD.encode(commit.as_bytes())),
            );
            self.call(target.node.as_ref(), "fs.put_small", args, 30.0)
                .await?;
        }
        self.set(
            did,
            &[
                ("state", text("completed")),
                ("done_bytes", SqlValue::Integer(total)),
                (
                    "files_done",
                    SqlValue::Integer(i64::try_from(items.len()).unwrap_or(0)),
                ),
                ("current_file", SqlValue::Null),
                ("current_dest", SqlValue::Null),
            ],
        );
        self.publish(did, 0.0, None);
        // the node finds the new model on its next scan; ask for it now so the library shows it at once
        let _ = self.library.rescan(Some(&target.host_id)).await;
        Ok(())
    }

    async fn link(&self, target: &Target, it: &Item) -> RunResult<()> {
        let Some((link, link_target)) = &it.link else {
            return Ok(());
        };
        let mut args = Obj::new();
        args.insert("root".into(), Value::String(target.root_id.clone()));
        args.insert("link".into(), Value::String(link.clone()));
        args.insert("target".into(), Value::String(link_target.clone()));
        match self
            .call(target.node.as_ref(), "fs.symlink", args, 30.0)
            .await
        {
            Ok(_) => Ok(()),
            Err(RunErr::Node(e)) if e.code == "conflict" || e.code == "io_error" => {
                // the link of an earlier run is still there: fine, as long as it is
                let st = self
                    .stat(
                        target.node.as_ref(),
                        &target.root_id,
                        std::slice::from_ref(link),
                    )
                    .await?;
                if st.get(link).is_some_and(|s| s.exists) {
                    Ok(())
                } else {
                    Err(RunErr::Node(e))
                }
            }
            Err(e) => Err(e),
        }
    }

    /// One file: from its source into `fs.write`, resuming a partial the node holds.
    async fn transfer(
        &self,
        target: &Target,
        it: &Item,
        partial: i64,
        progress: &mut Progress,
    ) -> RunResult<()> {
        let mut start = partial;
        let mut source = if start >= it.size {
            Source::Empty // the partial is whole: only the node's verification is left
        } else if let Some((repo, commit, path)) = &it.from_hf {
            let stream = self.hub.open_file(repo, commit, path, start).await?;
            if stream.offset() != start {
                // the server ignored the Range: this file starts over
                progress.done -= start - stream.offset();
                start = stream.offset();
            }
            Source::Hf(stream)
        } else if let Some((host, root, rel)) = &it.from_node {
            let node = self.fleet.node(host)?;
            Source::Node(NodeChunks::open(node.as_ref(), root, rel, start))
        } else {
            return Err(
                problem(500, "internal_error", "internal: a plan item has no source").into(),
            );
        };
        let res = self
            .write_remote(target, it, start, &mut source, progress)
            .await;
        source.close().await;
        res
    }

    async fn write_remote(
        &self,
        target: &Target,
        it: &Item,
        offset: i64,
        source: &mut Source,
        progress: &mut Progress,
    ) -> RunResult<()> {
        let mut args = Obj::new();
        args.insert("root".into(), Value::String(target.root_id.clone()));
        args.insert("rel".into(), Value::String(it.dest.clone()));
        args.insert("size".into(), Value::from(it.size));
        args.insert("offset".into(), Value::from(offset));
        if let Some(sha) = it.sha256.as_ref().filter(|s| !s.is_empty()) {
            args.insert("sha256".into(), Value::String(sha.clone()));
        }
        let mut stream = target.node.open("fs.write", Some(args));
        let stall = Duration::from_secs_f64(STALL_S);
        let timeout_err = || {
            RunErr::Node(NodeError::new(
                &target.host_id,
                "timeout",
                "the machine stopped accepting data; resume to continue",
            ))
        };
        let result: RunResult<()> = async {
            let mut sent = offset;
            while let Some(chunk) = source.next().await? {
                let n = i64::try_from(chunk.len()).unwrap_or(i64::MAX);
                if sent + n > it.size {
                    // never trust a source to stop at the declared size
                    return Err(problem(
                        502,
                        "size_mismatch",
                        format!(
                            "{}: the source sent more than the declared {} bytes",
                            py::take(&it.path, 80),
                            it.size
                        ),
                    )
                    .into());
                }
                for part in chunk.chunks(MAX_DATA) {
                    match tokio::time::timeout(stall, stream.write(part)).await {
                        Ok(r) => r?,
                        Err(_) => return Err(timeout_err()),
                    }
                }
                sent += n;
                self.advance(progress, n);
            }
            if sent != it.size {
                // no end(): the node would discard a short file; cancelling keeps the partial for the next try
                return Err(problem(
                    502,
                    "size_mismatch",
                    format!(
                        "{}: the source ended after {sent} of {} bytes",
                        py::take(&it.path, 80),
                        it.size
                    ),
                )
                .into());
            }
            match tokio::time::timeout(stall, stream.end()).await {
                Ok(r) => r?,
                Err(_) => return Err(timeout_err()),
            }
            #[allow(clippy::cast_precision_loss)]
            let verify = Duration::from_secs_f64(STALL_S + it.size as f64 / VERIFY_BPS);
            match tokio::time::timeout(verify, stream.result()).await {
                Ok(r) => {
                    r?;
                }
                Err(_) => return Err(timeout_err()),
            }
            Ok(())
        }
        .await;
        if result.is_err() {
            let _ = tokio::time::timeout(Duration::from_secs(5), stream.cancel()).await;
        }
        result
    }
}

/// Where one file's bytes come from.
enum Source {
    Empty,
    Hf(Box<dyn ByteStream>),
    Node(NodeChunks),
}

impl Source {
    async fn next(&mut self) -> RunResult<Option<Bytes>> {
        match self {
            Self::Empty => Ok(None),
            Self::Hf(s) => Ok(s.chunk().await?),
            Self::Node(n) => n.next().await,
        }
    }
    async fn close(&mut self) {
        match self {
            Self::Empty => {}
            Self::Hf(s) => s.close().await,
            Self::Node(n) => n.close().await,
        }
    }
}

/// `_node_chunks`: the DATA of an `fs.read`; a failure on the source machine surfaces at the end.
struct NodeChunks {
    stream: Box<dyn NodeStream>,
    done: bool,
}

impl NodeChunks {
    fn open(node: &dyn NodeClient, root: &str, rel: &str, offset: i64) -> Self {
        let mut args = Obj::new();
        args.insert("root".into(), Value::String(root.to_string()));
        args.insert("rel".into(), Value::String(rel.to_string()));
        args.insert("offset".into(), Value::from(offset));
        Self {
            stream: node.open("fs.read", Some(args)),
            done: false,
        }
    }

    async fn next(&mut self) -> RunResult<Option<Bytes>> {
        if self.done {
            return Ok(None);
        }
        loop {
            match self.stream.next().await {
                Some(StreamItem::Data(d)) if !d.is_empty() => return Ok(Some(d)),
                Some(_) => {}
                None => {
                    self.done = true;
                    return match self.stream.result().await {
                        Ok(_) => Ok(None),
                        Err(e) => {
                            self.cancel().await;
                            Err(e.into())
                        }
                    };
                }
            }
        }
    }

    async fn cancel(&mut self) {
        let _ = tokio::time::timeout(Duration::from_secs(5), self.stream.cancel()).await;
    }

    async fn close(&mut self) {
        if !self.done {
            self.cancel().await;
            self.done = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_helpers() {
        assert_eq!(hf_repo_rel("hf-home", "org/m"), "hub/models--org--m");
        assert_eq!(hf_repo_rel("hf-hub", "org/m"), "models--org--m");
        let (link, target) = snapshot_link("hub/models--o--m", "c", "sub/dir/f", "b");
        assert_eq!(link, "hub/models--o--m/snapshots/c/sub/dir/f");
        assert_eq!(target, "../../../../blobs/b");
        assert!(path_matches("a[1].gguf", Some(&["a[1].gguf".to_string()])));
        assert!(!safe_rel_path("a//b") && !safe_rel_path("..") && safe_rel_path("a/b"));
    }
}
