//! Python: backend/src/engine_console/services/bench.py
//! (tests/core/test_nodehttp_chat_bench.py).
//!
//! One-click benchmark against a ready instance, talking to its OpenAI endpoint through its host's node: TTFT / ITL /
//! e2e from a streaming client, decode and prefill tok/s from usage counts, throughput per concurrency level, and an
//! optional prefix-cache (cold vs warm TTFT) probe. Rows in `bench_runs`, progress on the bus topic "bench:<id>"
//! (events {"id", "event", ...}; the last one is "done"). The routes use the methods below.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::domain::errors::{AppResult, bad_request, conflict, not_found};
use crate::domain::models::{BenchRun, Page};
use crate::services::common::{EventBus, new_id, now, percentile};
use crate::services::gguf_info::py;
use crate::services::lifecycle::LifecycleService;
use crate::services::nodehttp::{HttpOpts, JSON_HEADERS, NodeHttp};
use crate::services::store::{SqlRow, Store, paginate};

/// hard ceilings: a bench suite must not become a self-inflicted DoS
pub const MAX_CONCURRENCY: i64 = 64;
pub const MAX_LEVELS: usize = 8;
/// roughly 14 tokens
const FILLER: &str = "The quick brown fox jumps over the lazy dog while counting to ten. ";

/// The preset suites, in Python's key order.
pub fn suites() -> Vec<(&'static str, Map<String, Value>)> {
    let obj = |v: Value| v.as_object().cloned().unwrap_or_default();
    vec![
        (
            "quick",
            obj(
                json!({"concurrency": [1, 4], "prompt_tokens": 512, "max_tokens": 128, "rounds": 1, "prefix_cache": false}),
            ),
        ),
        (
            "standard",
            obj(
                json!({"concurrency": [1, 4, 16, 64], "prompt_tokens": 1024, "max_tokens": 256, "rounds": 2,
                       "prefix_cache": true}),
            ),
        ),
    ]
}

/// One streamed request's measurements.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReqStat {
    pub ttft: Option<f64>,
    pub e2e: f64,
    pub itl: Vec<f64>,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub error: Option<String>,
}

pub fn make_prompt(approx_tokens: i64, nonce: Option<&str>) -> String {
    let n = usize::try_from(py::floordiv(approx_tokens, 14).unwrap_or(0).max(1)).unwrap_or(1);
    let body = FILLER.repeat(n);
    match nonce.filter(|x| !x.is_empty()) {
        Some(nc) => format!("[{nc}] {body}\nSummarise the text above in one sentence."),
        None => format!("{body}\nSummarise the text above in one sentence."),
    }
}

/// `int(x)` of a usage count.
fn py_int(v: Option<&Value>) -> i64 {
    match v {
        Some(Value::Number(n)) => n.as_i64().unwrap_or_else(|| {
            #[allow(clippy::cast_possible_truncation)]
            let x = n.as_f64().unwrap_or(0.0).trunc() as i64;
            x
        }),
        Some(Value::Bool(b)) => i64::from(*b),
        Some(Value::String(s)) => crate::services::store::py_int(s).unwrap_or(0),
        _ => 0,
    }
}

/// One streamed chat request to an engine through its host's node, timed from the first byte request to the last.
pub async fn one_request(
    http: &NodeHttp,
    host_id: &str,
    port: i64,
    model: &str,
    prompt: &str,
    max_tokens: i64,
) -> ReqStat {
    let mut st = ReqStat::default();
    let body = crate::pyjson::dumps(
        &json!({"model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.7}),
    );
    let mut headers: Vec<(String, String)> = JSON_HEADERS
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    if let Some(a) = headers.iter_mut().find(|(k, _)| k == "accept") {
        a.1 = "text/event-stream".into();
    }
    let t0 = Instant::now();
    let mut last = t0;
    let opened = http
        .open(
            host_id,
            port,
            "POST",
            "/v1/chat/completions",
            HttpOpts {
                body: Some(body.into_bytes()),
                headers,
                timeout_s: Some(300.0),
                max_bytes: None,
            },
        )
        .await;
    let mut resp = match opened {
        Ok(r) => r,
        Err(e) => {
            // the node refused, timed out or could not reach the engine
            st.error = Some(e.code().to_string());
            st.e2e = t0.elapsed().as_secs_f64();
            return st;
        }
    };
    if resp.status != 200 {
        st.error = Some(format!("HTTP {}", resp.status));
        resp.close().await;
        return st; // Python returns inside the try: e2e is never measured
    }
    loop {
        match resp.line().await {
            Ok(Some(line)) => {
                if !line.starts_with("data:") || py::strip(&line) == "data: [DONE]" {
                    continue;
                }
                let Ok(Value::Object(ev)) = serde_json::from_str::<Value>(&line[5..]) else {
                    continue;
                };
                if let Some(Value::Object(u)) = ev.get("usage").filter(|u| py::truthy(u)) {
                    st.prompt_tokens = py_int(u.get("prompt_tokens"));
                    st.completion_tokens = py_int(u.get("completion_tokens"));
                }
                let first = match ev.get("choices") {
                    Some(Value::Array(a)) if !a.is_empty() => a[0].clone(),
                    _ => json!({}),
                };
                let ch = first.get("delta").cloned().unwrap_or_else(|| json!({}));
                let any = ["content", "reasoning_content", "reasoning"]
                    .iter()
                    .any(|k| ch.get(*k).is_some_and(py::truthy));
                if any {
                    let t = Instant::now();
                    if st.ttft.is_none() {
                        st.ttft = Some((t - t0).as_secs_f64());
                    } else {
                        st.itl.push((t - last).as_secs_f64());
                    }
                    last = t;
                }
            }
            Ok(None) => break,
            Err(e) => {
                st.error = Some(e.code().to_string());
                break;
            }
        }
    }
    resp.close().await;
    st.e2e = t0.elapsed().as_secs_f64();
    st
}

#[allow(clippy::cast_precision_loss)]
fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().fold(0.0, |a, b| a + b) / v.len() as f64)
}

/// `summarize(stats)`: request counts and the percentiles of one level.
#[allow(clippy::cast_precision_loss)]
pub fn summarize(stats: &[ReqStat]) -> Map<String, Value> {
    let ok: Vec<&ReqStat> = stats
        .iter()
        .filter(|s| s.error.is_none() && s.ttft.is_some())
        .collect();
    let ttft: Vec<f64> = ok.iter().filter_map(|s| s.ttft).collect();
    let e2e: Vec<f64> = ok.iter().map(|s| s.e2e).collect();
    let itl: Vec<f64> = ok.iter().flat_map(|s| s.itl.iter().copied()).collect();
    let dec: Vec<f64> = ok
        .iter()
        .filter_map(|s| {
            let t = s.ttft?;
            (s.e2e > t && s.completion_tokens != 0)
                .then(|| s.completion_tokens as f64 / (s.e2e - t))
        })
        .collect();
    let pre: Vec<f64> = ok
        .iter()
        .filter_map(|s| {
            let t = s.ttft.filter(|t| *t != 0.0)?;
            (s.prompt_tokens != 0).then(|| s.prompt_tokens as f64 / t)
        })
        .collect();
    let opt = |x: Option<f64>| x.map_or(Value::Null, Value::from);
    json!({"requests": stats.len(), "errors": stats.len() - ok.len(),
           "ttft_p50_s": opt(percentile(&ttft, 50.0)), "ttft_p95_s": opt(percentile(&ttft, 95.0)),
           "itl_p50_s": opt(percentile(&itl, 50.0)), "itl_p95_s": opt(percentile(&itl, 95.0)),
           "e2e_p50_s": opt(percentile(&e2e, 50.0)), "e2e_p95_s": opt(percentile(&e2e, 95.0)),
           "decode_tps": opt(mean(&dec)), "prefill_tps": opt(mean(&pre))})
    .as_object()
    .cloned()
    .unwrap_or_default()
}

/// `isinstance(v, int)` (a bool is an int in Python) as an i64.
fn int_like(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(i64::from(*b)),
        _ => py::as_int(v),
    }
}

struct Task {
    handle: JoinHandle<()>,
    token: CancellationToken,
}

struct Inner {
    db: Arc<Store>,
    lifecycle: Arc<LifecycleService>,
    http: Arc<NodeHttp>,
    bus: Arc<EventBus>,
    tasks: Mutex<HashMap<String, Task>>,
}

pub struct BenchService {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for BenchService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BenchService")
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn row_of(r: &SqlRow) -> BenchRun {
    let obj = |col: &str| match r.json_text(col) {
        Value::Object(o) => o,
        _ => Map::new(),
    };
    BenchRun {
        id: r.str_or("id", ""),
        instance_id: r.str_or("instance_id", ""),
        engine: r.text("engine"),
        repo_id: r.text("repo_id"),
        profile_id: r.text("profile_id"),
        params: obj("params"),
        suite: obj("suite"),
        state: r.str_or("state", ""),
        results: obj("results"),
        error: r.text("error"),
        created_at: r.f64("created_at").unwrap_or(0.0),
        finished_at: r.f64("finished_at"),
    }
}

impl BenchService {
    pub fn new(
        db: Arc<Store>,
        lifecycle: Arc<LifecycleService>,
        http: Arc<NodeHttp>,
        bus: Arc<EventBus>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                db,
                lifecycle,
                http,
                bus,
                tasks: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// `resolve_suite`: a preset by name, or a custom dict merged over "quick" and clamped.
    pub fn resolve_suite(suite: &Value) -> AppResult<Map<String, Value>> {
        let presets = suites();
        if let Value::String(name) = suite {
            return match presets.iter().find(|(n, _)| n == name) {
                Some((_, s)) => Ok(s.clone()),
                None => {
                    let mut names: Vec<&str> = presets.iter().map(|(n, _)| *n).collect();
                    names.sort_unstable();
                    Err(bad_request(format!("unknown suite '{name}'"))
                        .code("unknown_suite")
                        .with("available", Value::from(names))
                        .into())
                }
            };
        }
        let quick = presets[0].1.clone();
        let custom = suite.as_object().cloned().unwrap_or_default();
        let mut merged = quick.clone();
        for (k, v) in &custom {
            merged.insert(k.clone(), v.clone());
        }
        let mut unknown: Vec<String> = custom
            .keys()
            .filter(|k| !quick.contains_key(*k))
            .cloned()
            .collect();
        if !unknown.is_empty() {
            unknown.sort();
            let repr = ec_proto::pyfmt::repr_value(&Value::from(unknown));
            return Err(bad_request(format!("unknown suite fields: {repr}"))
                .code("invalid_suite")
                .into());
        }
        let conc_ok = match merged.get("concurrency") {
            Some(Value::Array(c)) => {
                (1..=MAX_LEVELS).contains(&c.len())
                    && c.iter()
                        .all(|x| py::as_int(x).is_some_and(|n| (1..=MAX_CONCURRENCY).contains(&n)))
            }
            _ => false,
        };
        if !conc_ok {
            return Err(bad_request(format!(
                "concurrency must be 1..{MAX_LEVELS} ints in 1..{MAX_CONCURRENCY}"
            ))
            .code("invalid_suite")
            .into());
        }
        for (k, lo, hi) in [
            ("prompt_tokens", 1, 32_768),
            ("max_tokens", 1, 4_096),
            ("rounds", 1, 5),
        ] {
            let ok = merged
                .get(k)
                .and_then(int_like)
                .is_some_and(|v| (lo..=hi).contains(&v));
            if !ok {
                return Err(bad_request(format!("{k} must be an int in {lo}..{hi}"))
                    .code("invalid_suite")
                    .into());
            }
        }
        Ok(merged)
    }

    /// Start-up: runs left "running" by a previous process become failed.
    pub fn recover(&self) -> AppResult<()> {
        self.inner.db.execute(
            "UPDATE bench_runs SET state='failed', error='interrupted: the console stopped', finished_at=? \
             WHERE state='running'",
            &[&now()],
        )?;
        Ok(())
    }

    pub async fn shutdown(&self) {
        let tasks: Vec<Task> = lock(&self.inner.tasks).drain().map(|(_, t)| t).collect();
        for t in &tasks {
            t.token.cancel();
        }
        for t in tasks {
            let _ = t.handle.await;
        }
        let _ = self.recover();
    }

    pub fn get(&self, bid: &str) -> AppResult<BenchRun> {
        match self
            .inner
            .db
            .one("SELECT * FROM bench_runs WHERE id=?", &[&bid])?
        {
            Some(r) => Ok(row_of(&r)),
            None => Err(not_found(format!("no such benchmark: {bid}")).into()),
        }
    }

    pub fn list_page(
        &self,
        limit: i64,
        cursor: Option<&str>,
        instance_id: Option<&str>,
    ) -> AppResult<Page<BenchRun>> {
        let (limit, off) = paginate(limit, cursor);
        let rows = match instance_id.filter(|i| !i.is_empty()) {
            Some(iid) => self.inner.db.all(
                "SELECT * FROM bench_runs WHERE instance_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                &[&iid, &(limit + 1), &off],
            )?,
            None => self.inner.db.all(
                "SELECT * FROM bench_runs  ORDER BY created_at DESC LIMIT ? OFFSET ?",
                &[&(limit + 1), &off],
            )?,
        };
        let lim = usize::try_from(limit).unwrap_or(0);
        Ok(Page::new(
            rows.iter().take(lim).map(row_of).collect(),
            (rows.len() > lim).then(|| (off + limit).to_string()),
        ))
    }

    /// `suite` is a preset name or a custom dict; `confirm_external` is required for engines the console does not own.
    pub async fn start(
        &self,
        instance_id: &str,
        suite: &Value,
        confirm_external: bool,
    ) -> AppResult<BenchRun> {
        let inner = &self.inner;
        let inst = inner.lifecycle.get(instance_id)?;
        if inst.state != "ready" || inst.port.is_none_or(|p| p == 0) {
            return Err(conflict(format!(
                "instance is {}; benchmarks need a ready instance",
                inst.state
            ))
            .code("instance_not_ready")
            .into());
        }
        let mut spec = Self::resolve_suite(suite)?;
        if !inst.managed {
            if !confirm_external {
                return Err(bad_request(
                    "this engine is not owned by the console: a benchmark sends real load to it. Repeat the request \
                     with confirm_external=true to proceed",
                )
                .code("confirm_external_required")
                .into());
            }
            spec.insert(
                "note".into(),
                Value::String(
                    "load was sent to an external engine the console does not own".into(),
                ),
            );
        }
        let bid = new_id("bench_");
        inner.db.execute(
            "INSERT INTO bench_runs(id,instance_id,engine,repo_id,profile_id,params,suite,state,results,created_at) \
             VALUES(?,?,?,?,?,?,?,?,?,?)",
            &[
                &bid,
                &instance_id,
                &inst.engine,
                &inst.repo_id,
                &inst.profile_id,
                &crate::pyjson::dumps(&Value::Object(inst.params.clone())),
                &crate::pyjson::dumps(&Value::Object(spec.clone())),
                &"running",
                &"{}",
                &now(),
            ],
        )?;
        let row = self.get(&bid)?;
        let token = CancellationToken::new();
        let (me, b2, t2) = (Arc::clone(inner), bid.clone(), token.clone());
        let (host, port, model) = (inst.host_id.clone(), inst.port.unwrap_or(0), inst.repo_id);
        let handle =
            tokio::spawn(async move { Inner::run(me, b2, host, port, model, spec, t2).await });
        lock(&inner.tasks).insert(bid, Task { handle, token });
        Ok(row)
    }

    /// Test helper: block until the run's task ends.
    pub async fn wait(&self, bid: &str) {
        loop {
            let done = lock(&self.inner.tasks)
                .get(bid)
                .is_none_or(|t| t.handle.is_finished());
            if done {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
}

impl Inner {
    fn emit(&self, bid: &str, ev: Value) {
        let mut o = Map::new();
        o.insert("id".into(), Value::String(bid.to_string()));
        if let Value::Object(e) = ev {
            o.extend(e);
        }
        self.bus.publish(&format!("bench:{bid}"), Value::Object(o));
    }

    #[allow(clippy::too_many_arguments)]
    async fn run(
        me: Arc<Self>,
        bid: String,
        host: String,
        port: i64,
        model: String,
        spec: Map<String, Value>,
        token: CancellationToken,
    ) {
        let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let results = Arc::new(Mutex::new(json!({"concurrency": []})));
        let work = {
            let (me, started, results, spec, host, model, bid2) = (
                Arc::clone(&me),
                Arc::clone(&started),
                Arc::clone(&results),
                spec.clone(),
                host.clone(),
                model.clone(),
                bid.clone(),
            );
            async move {
                me.levels(&bid2, &host, port, &model, &spec, &started, &results)
                    .await
            }
        };
        // a panic in the run is Python's `except Exception` around `_run`: the run is marked failed
        let work = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(work));
        let res = tokio::select! {
            biased;
            () = token.cancelled() => None,
            r = work => Some(r.unwrap_or_else(|p| Err(format!(
                "panic: {}",
                crate::panics::redacted(&crate::panics::payload_text(&*p))
            )))),
        };
        match res {
            None => {
                // a task cancelled before it ran leaves the row running (shutdown's recover() closes it);
                // one cancelled mid-run is closed as cancelled, and no "done" event follows (Python re-raises)
                if started.load(std::sync::atomic::Ordering::SeqCst) {
                    let _ = me.db.execute(
                        "UPDATE bench_runs SET state='failed', error='cancelled', finished_at=? WHERE id=?",
                        &[&now(), &bid],
                    );
                }
                return;
            }
            Some(Ok(())) => {
                let text = crate::pyjson::dumps(&lock(&results));
                let _ = me.db.execute(
                    "UPDATE bench_runs SET state='completed', results=?, finished_at=? WHERE id=?",
                    &[&text, &now(), &bid],
                );
            }
            Some(Err(msg)) => {
                tracing::error!(target: "engine_console.services.bench", "bench {} failed", bid);
                let text = crate::pyjson::dumps(&lock(&results));
                let _ = me.db.execute(
                    "UPDATE bench_runs SET state='failed', error=?, results=?, finished_at=? WHERE id=?",
                    &[&msg, &text, &now(), &bid],
                );
            }
        }
        let state = me
            .db
            .one("SELECT state FROM bench_runs WHERE id=?", &[&bid])
            .ok()
            .flatten()
            .map(|r| r.str_or("state", ""))
            .unwrap_or_default();
        me.emit(&bid, json!({"event": "done", "state": state}));
    }

    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    async fn levels(
        &self,
        bid: &str,
        host: &str,
        port: i64,
        model: &str,
        spec: &Map<String, Value>,
        started: &std::sync::atomic::AtomicBool,
        results: &Mutex<Value>,
    ) -> Result<(), String> {
        started.store(true, std::sync::atomic::Ordering::SeqCst);
        let conc: Vec<i64> = spec
            .get("concurrency")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(py::as_int).collect())
            .unwrap_or_default();
        let get = |k: &str| spec.get(k).and_then(int_like).unwrap_or(0);
        let (rounds, prompt_tokens, max_tokens) =
            (get("rounds"), get("prompt_tokens"), get("max_tokens"));
        for c in conc {
            self.emit(bid, json!({"event": "level_start", "concurrency": c}));
            let mut stats: Vec<ReqStat> = Vec::new();
            let t0 = Instant::now();
            for _ in 0..rounds.max(0) {
                let futs =
                    (0..c.max(0)).map(|_| {
                        let nonce = nonce8();
                        let prompt = make_prompt(prompt_tokens, Some(&nonce));
                        async move {
                            one_request(&self.http, host, port, model, &prompt, max_tokens).await
                        }
                    });
                stats.extend(futures::future::join_all(futs).await);
            }
            let wall = t0.elapsed().as_secs_f64();
            let mut level = Map::new();
            level.insert("concurrency".into(), Value::from(c));
            level.extend(summarize(&stats));
            let tokens: i64 = stats.iter().map(|s| s.completion_tokens).sum();
            level.insert(
                "throughput_tps".into(),
                if wall > 0.0 {
                    Value::from(tokens as f64 / wall)
                } else {
                    Value::Null
                },
            );
            {
                let mut r = lock(results);
                if let Some(Value::Array(a)) = r.get_mut("concurrency") {
                    a.push(Value::Object(level.clone()));
                }
                if c == 1 {
                    let mut single = level.clone();
                    single.shift_remove("concurrency");
                    r["single"] = Value::Object(single);
                }
            }
            self.emit(bid, json!({"event": "level_done", "level": level}));
        }
        if spec.get("prefix_cache").is_some_and(py::truthy) {
            let prompt = make_prompt(prompt_tokens, Some(&nonce8()));
            let cold = one_request(&self.http, host, port, model, &prompt, 8).await;
            let warm = one_request(&self.http, host, port, model, &prompt, 8).await;
            if let (Some(c), Some(w)) = (
                cold.ttft.filter(|x| *x != 0.0),
                warm.ttft.filter(|x| *x != 0.0),
            ) {
                lock(results)["prefix_cache"] =
                    json!({"cold_ttft_s": c, "warm_ttft_s": w, "speedup": c / w});
            }
        }
        Ok(())
    }
}

/// `uuid.uuid4().hex[:8]`
fn nonce8() -> String {
    hex::encode(crate::services::common::random_bytes(4))
}
