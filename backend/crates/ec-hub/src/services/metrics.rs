//! Python: backend/src/engine_console/services/metrics.py
//! (tests/core/test_metrics_fleet.py, test_metrics_prune.py).
//!
//! Metrics (contract v2, §10.9). The hub sends each node `metrics.configure` (the metric-name prefixes of every
//! engine it knows); the node scrapes its engine endpoints and PUSHES `metrics` messages, which the fleet
//! re-publishes on the EventBus as topic `node.metrics` with payload {"host_id", "msg"}. Each one is mapped to an
//! instance (by host + port or container), parsed with its adapter (or the generic map for engines without one) and
//! kept in a ring buffer (15 min) plus SQLite rollups (1 h / 1 d). Per-host system series (RAM, shared memory, swap)
//! come from the node's samples.
//!
//! A managed engine whose node pushed nothing for a while is also read directly through `http.request` (the pull
//! fallback). Points are stamped with the hub's clock, never the node's.
//!
//! Review M7 (security control, named tests in tests/test_metrics_prune.rs): state keyed by names a node chooses is
//! bounded per host (MAX_KEYS_PER_HOST, the stalest series goes first, never another host's), expires after
//! KEY_TTL_S, and rollup rows age out (14 d hourly, 400 d daily). Review L3: non-finite values never reach a series.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rusqlite::types::Value as SqlValue;
use serde_json::{Map, Value, json};

use crate::adapters::AdapterRegistry;
use crate::domain::errors::{AppResult, Obj, Problem, bad_request};
use crate::domain::fleet::{FleetPort, LOCAL};
use crate::domain::models::{Instance, MetricPoint};
use crate::services::common::EventBus;
use crate::services::discovery::parse_generic_metrics;
use crate::services::hostmem::meminfo_from_sample;
use crate::services::lifecycle::LifecycleService;
use crate::services::nodeerr::NodeCaller;
use crate::services::nodehttp::{HttpOpts, NodeHttp};
use crate::services::nodeview as nv;
use crate::services::store::Store;
use crate::sync::Recover as _;

pub const RING_SECONDS: i64 = 15 * 60;
pub const TOPIC: &str = "metrics";
/// the console's own machine keeps its v1 key, so its history survives the upgrade
pub const SYSTEM_KEY: &str = "system";
pub const PUSH_TOPIC: &str = "node.metrics";
/// distinct series per host; past it the least recently updated one is evicted
pub const MAX_KEYS_PER_HOST: usize = 256;
/// in-memory state of a series nobody updated for this long is dropped (the ring holds 15 min)
pub const KEY_TTL_S: f64 = 3600.0;
/// rollup rows older than this are deleted (the 1 h series serves windows up to 3 d)
pub const HOURLY_KEEP_S: f64 = 14.0 * 86400.0;
pub const DAILY_KEEP_S: f64 = 400.0 * 86400.0;
/// a prune pass is O(series) in memory and one indexed delete per resolution
pub const PRUNE_EVERY_S: f64 = 300.0;
/// Metric-name prefixes per engine: what the node keeps from a scrape (adapters may declare their own).
pub const PREFIXES: &[(&str, &[&str])] = &[
    ("vllm", &["vllm:"]),
    ("sglang", &["sglang:"]),
    ("llamacpp", &["llamacpp:"]),
    ("trtllm", &["trtllm", "tensorrt_llm"]),
    ("dynamo", &["dynamo_"]),
];

/// "15m" / "1h" / "6h" / "1d" / "7d" -> seconds; 400 invalid_window otherwise.
pub fn parse_window(w: &str) -> Result<i64, Problem> {
    let bad = || bad_request("window must look like 15m, 1h, 6h, 1d, 7d").code("invalid_window");
    let mut chars = w.chars();
    let unit = chars.next_back().ok_or_else(bad)?;
    let digits = chars.as_str();
    let mult = match unit {
        's' => 1,
        'm' => 60,
        'h' => 3600,
        'd' => 86400,
        _ => return Err(bad()),
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return Err(bad());
    }
    digits
        .parse::<i64>()
        .ok()
        .and_then(|n| n.checked_mul(mult))
        .ok_or_else(bad)
}

/// The adapter's metrics parser as the poller and the consumer (non-HTTP callers) must call it: the `try_*` form
/// (PORTING §A5), with a raised exception as `Kind: message`.
fn try_parse_metrics(
    adapter: &dyn crate::adapters::base::EngineAdapter,
    text: &str,
) -> Result<crate::adapters::base::FloatMap, String> {
    adapter.try_parse_metrics(text).map_err(|e| e.to_string())
}

pub fn system_key(host_id: &str) -> String {
    if host_id == LOCAL {
        SYSTEM_KEY.to_string()
    } else {
        format!("{SYSTEM_KEY}:{host_id}")
    }
}

pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

#[derive(Default)]
struct State {
    ring: HashMap<String, VecDeque<MetricPoint>>,
    last_counters: HashMap<String, (f64, Vec<(String, f64)>)>,
    /// instance id -> when its engine metrics last arrived
    last_push: HashMap<String, f64>,
    last_pull: HashMap<String, f64>,
    /// host id -> the sample `seq` already recorded
    last_seq: HashMap<String, Value>,
    configured: HashMap<String, (bool, Option<f64>)>,
    warned: Vec<(String, Option<f64>)>,
    /// host id -> series key -> when it was last updated (hub clock), in insertion order
    keys: HashMap<String, Vec<(String, f64)>>,
    next_prune: f64,
    adopted: bool,
}

pub struct MetricsService {
    db: Arc<Store>,
    lifecycle: Arc<LifecycleService>,
    adapters: AdapterRegistry,
    fleet: Arc<dyn FleetPort>,
    nodes: NodeCaller,
    http: Arc<NodeHttp>,
    bus: Arc<EventBus>,
    interval_s: f64,
    ring_cap: usize,
    max_keys: AtomicUsize,
    clock: Mutex<Clock>,
    st: Mutex<State>,
}

impl std::fmt::Debug for MetricsService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MetricsService")
    }
}

fn point_json(pt: &MetricPoint) -> Map<String, Value> {
    match serde_json::to_value(pt) {
        Ok(Value::Object(o)) => o,
        _ => Map::new(),
    }
}

/// `dict.update` on an ordered float map.
fn put(vals: &mut Vec<(String, f64)>, k: &str, v: f64) {
    if let Some(slot) = vals.iter_mut().find(|(kk, _)| kk == k) {
        slot.1 = v;
    } else {
        vals.push((k.to_string(), v));
    }
}

fn get(vals: &[(String, f64)], k: &str) -> Option<f64> {
    vals.iter().find(|(kk, _)| kk == k).map(|(_, v)| *v)
}

impl MetricsService {
    pub fn new(
        db: Arc<Store>,
        lifecycle: Arc<LifecycleService>,
        adapters: AdapterRegistry,
        fleet: Arc<dyn FleetPort>,
        http: Arc<NodeHttp>,
        bus: Arc<EventBus>,
        interval_s: f64,
    ) -> Self {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let ring_cap = (RING_SECONDS as f64 / interval_s.max(0.5)) as usize + 1;
        Self {
            db,
            lifecycle,
            adapters,
            nodes: NodeCaller::new(Arc::clone(&fleet)),
            fleet,
            http,
            bus,
            interval_s,
            ring_cap,
            max_keys: AtomicUsize::new(MAX_KEYS_PER_HOST),
            clock: Mutex::new(Arc::new(crate::services::common::now)),
            st: Mutex::new(State::default()),
        }
    }

    /// Replace the clock (tests: Python sets `metrics._clock`).
    pub fn set_clock(&self, clock: Clock) {
        *self.clock.guard() = clock;
    }

    /// The per-host series cap (tests: Python monkeypatches `metrics.MAX_KEYS_PER_HOST`).
    pub fn set_max_keys_per_host(&self, n: usize) {
        self.max_keys.store(n, Ordering::SeqCst);
    }

    fn now(&self) -> f64 {
        (*self.clock.guard())()
    }

    fn st(&self) -> std::sync::MutexGuard<'_, State> {
        self.st.guard()
    }

    // -- test views of the bounded state (Python's tests read the private dicts) -------------------------------------
    /// Number of series a host holds in memory (None: no bookkeeping for that host).
    pub fn keys_of(&self, host_id: &str) -> Option<usize> {
        self.st().keys.get(host_id).map(Vec::len)
    }

    /// Number of ring buffers in memory.
    pub fn ring_len(&self) -> usize {
        self.st().ring.len()
    }

    /// Whether counters of a series are remembered (rate derivation).
    pub fn has_counters(&self, key: &str) -> bool {
        self.st().last_counters.contains_key(key)
    }

    /// Forget the in-memory caps, as a restart does (the rollup rows stay).
    pub fn forget_memory(&self) {
        let mut st = self.st();
        st.keys.clear();
        st.adopted = false;
    }

    /// Seed per-host bookkeeping (tests of the prune of hosts that left hosts.json).
    pub fn seed_bookkeeping(&self, host_id: &str, seq: Value, configured: (bool, Option<f64>)) {
        let mut st = self.st();
        st.last_seq.insert(host_id.to_string(), seq);
        st.configured.insert(host_id.to_string(), configured);
    }

    pub fn seed_warned(&self, host_id: &str, connected_since: Option<f64>) {
        self.st()
            .warned
            .push((host_id.to_string(), connected_since));
    }

    /// (hosts with a remembered sample seq, hosts configured, warnings remembered)
    pub fn bookkeeping(&self) -> (Vec<String>, Vec<String>, usize) {
        let st = self.st();
        (
            st.last_seq.keys().cloned().collect(),
            st.configured.keys().cloned().collect(),
            st.warned.len(),
        )
    }

    // -- node configuration -------------------------------------------------------------------------------------------
    /// engine id -> metric-name prefixes, for `metrics.configure` (at most 16 engines, 32 prefixes of 64 chars).
    pub fn families(&self) -> Vec<(String, Vec<String>)> {
        let mut out: Vec<(String, Vec<String>)> = PREFIXES
            .iter()
            .map(|(e, p)| {
                (
                    (*e).to_string(),
                    p.iter().map(|s| (*s).to_string()).collect(),
                )
            })
            .collect();
        for a in self.adapters.list() {
            let id = a.id().to_string();
            let prefixes = match a.metric_prefixes().filter(|p| !p.is_empty()) {
                Some(p) => p,
                None => out
                    .iter()
                    .find(|(e, _)| *e == id)
                    .map_or_else(|| vec![format!("{id}:")], |(_, p)| p.clone()),
            };
            if let Some(slot) = out.iter_mut().find(|(e, _)| *e == id) {
                slot.1 = prefixes;
            } else {
                out.push((id, prefixes));
            }
        }
        out.into_iter()
            .take(16)
            .filter(|(_, p)| !p.is_empty())
            .map(|(e, p)| {
                (
                    e,
                    p.into_iter()
                        .take(32)
                        .map(|s| s.chars().take(64).collect())
                        .collect(),
                )
            })
            .collect()
    }

    pub async fn configure(&self, host_id: &str) -> AppResult<()> {
        let families: Map<String, Value> = self
            .families()
            .into_iter()
            .map(|(e, p)| (e, json!(p)))
            .collect();
        let mut args = Obj::new();
        args.insert("interval_s".into(), json!(self.interval_s));
        args.insert("families".into(), Value::Object(families));
        self.nodes
            .call(host_id, "metrics.configure", Some(args), 20.0)
            .await?;
        Ok(())
    }

    /// Send `metrics.configure` to every host whose link just came up (or came back).
    pub async fn sync_hosts(&self) {
        for h in self.fleet.hosts() {
            let Ok(node) = self.fleet.node(&h.id) else {
                continue;
            };
            let key = (nv::link_up(node.as_ref()), node.link().connected_since);
            let prev = self.st().configured.get(&h.id).copied();
            if key.0 && Some(key) != prev && self.configure(&h.id).await.is_err() {
                let mut st = self.st();
                let w = (h.id.clone(), key.1);
                if !st.warned.contains(&w) {
                    // once per connection, not every round
                    st.warned.push(w);
                    tracing::warn!(target: "engine_console.services.metrics", "metrics.configure failed on {}", h.id);
                }
                continue; // try again next round
            }
            self.st().configured.insert(h.id.clone(), key);
        }
    }

    // -- routing a push to an instance ------------------------------------------------------------------------------
    /// `_parse`. `Err` is an exception the adapter's parser raised (§A5): the caller drops that scrape, as Python's
    /// consumer logs it and its pull fallback swallows it.
    fn parse(&self, engine: &str, text: &str) -> Result<Vec<(String, f64)>, String> {
        let vals = match self.adapters.get(engine) {
            Ok(a) => {
                let mut v: Vec<(String, f64)> = Vec::new();
                for (k, x) in try_parse_metrics(a.as_ref(), text)? {
                    put(&mut v, &k, x);
                }
                v
            }
            Err(_) => parse_generic_metrics(engine, text),
        };
        // Prometheus text allows +Inf/NaN; they would reach SSE as `Infinity`/`NaN`, which is not JSON (review L3)
        Ok(vals.into_iter().filter(|(_, v)| v.is_finite()).collect())
    }

    /// One pushed `metrics` message (already sanitised by the fleet): find its instance, parse, record.
    pub fn ingest(&self, host_id: &str, msg: &Obj) {
        let port = msg.get("port").and_then(nv::int_of);
        let container = match msg.get("container") {
            Some(Value::String(c)) if !c.is_empty() => Some(c.clone()),
            _ => None,
        };
        match msg.get("status") {
            Some(Value::Bool(_)) => return,
            Some(s) => {
                if let Some(code) = nv::int_of(s)
                    && !(200..300).contains(&code)
                {
                    return;
                }
            }
            None => {}
        }
        let Some(inst) = self.lifecycle.lookup(host_id, port, container.as_deref()) else {
            return;
        };
        let text: String = msg
            .get("text")
            .filter(|v| ec_proto::sanitize::truthy(v))
            .map_or_else(String::new, ec_proto::pyfmt::str_value)
            .chars()
            .take(2_000_000)
            .collect();
        let engine = if self.adapters.contains(&inst.engine)
            || PREFIXES.iter().any(|(e, _)| *e == inst.engine)
        {
            inst.engine.clone()
        } else {
            msg.get("engine")
                .filter(|v| ec_proto::sanitize::truthy(v))
                .map_or_else(|| inst.engine.clone(), ec_proto::pyfmt::str_value)
        };
        let vals = match self.parse(&engine, &text) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(target: "engine_console.services.metrics", "could not process a pushed metrics message: {}", e);
                return;
            }
        };
        self.record(&inst, vals, self.now(), true);
    }

    // -- recording ----------------------------------------------------------------------------------------------------
    fn record(&self, inst: &Instance, mut vals: Vec<(String, f64)>, t: f64, pushed: bool) {
        if let Some(mem) = &inst.host_memory {
            // cgroup-based host RAM, in the same series and ring buffer
            put(&mut vals, "host_ram_gib", mem.total_gib);
            put(&mut vals, "host_ram_anon_gib", mem.anon_gib);
            put(&mut vals, "host_ram_cache_gib", mem.cache_gib);
            put(&mut vals, "host_ram_shmem_gib", mem.shmem_gib);
        }
        if vals.is_empty() {
            return;
        }
        let touch = {
            let mut st = self.st();
            if pushed {
                st.last_push.insert(inst.id.clone(), t);
            }
            Self::derive_rates(&mut st, &inst.id, t, &mut vals)
        };
        if touch || get(&vals, "requests_running").unwrap_or(0.0) > 0.0 {
            self.lifecycle.touch(&inst.id);
        }
        let values: Map<String, Value> = vals.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
        let pt = MetricPoint { t, values };
        self.claim(&inst.host_id, &inst.id, t);
        self.push_ring(&inst.id, pt.clone());
        self.rollup(&inst.id, &pt, &inst.host_id);
        let mut ev = Map::new();
        ev.insert("instance_id".into(), json!(inst.id));
        ev.extend(point_json(&pt));
        self.bus.publish(TOPIC, Value::Object(ev));
    }

    fn push_ring(&self, key: &str, pt: MetricPoint) {
        let cap = self.ring_cap;
        let mut st = self.st();
        let ring = st.ring.entry(key.to_string()).or_default();
        if ring.len() == cap {
            ring.pop_front();
        }
        ring.push_back(pt);
    }

    // -- bounded state (review M7) ----------------------------------------------------------------------------------
    /// Note that `key` was updated; a host that already has MAX_KEYS_PER_HOST series loses its stalest one first.
    fn claim(&self, host_id: &str, key: &str, t: f64) {
        let max = self.max_keys.load(Ordering::SeqCst);
        let victim = {
            let mut st = self.st();
            let keys = st.keys.entry(host_id.to_string()).or_default();
            let mut victim = None;
            if !keys.iter().any(|(k, _)| k == key) && keys.len() >= max {
                let keep = system_key(host_id);
                let mut best: Option<(&String, f64)> = None;
                for (k, last) in keys.iter() {
                    if *k != keep && best.is_none_or(|(_, b)| *last < b) {
                        best = Some((k, *last));
                    }
                }
                victim = best.map(|(k, _)| k.clone());
            }
            victim
        };
        if let Some(v) = victim {
            self.forget(host_id, &v, true);
        }
        let mut st = self.st();
        let keys = st.keys.entry(host_id.to_string()).or_default();
        if let Some(slot) = keys.iter_mut().find(|(k, _)| k == key) {
            slot.1 = t;
        } else {
            keys.push((key.to_string(), t));
        }
    }

    fn forget(&self, host_id: &str, key: &str, rollups: bool) {
        {
            let mut st = self.st();
            if let Some(keys) = st.keys.get_mut(host_id) {
                keys.retain(|(k, _)| k != key);
            }
            st.ring.remove(key);
            st.last_counters.remove(key);
            st.last_push.remove(key);
            st.last_pull.remove(key);
        }
        if rollups {
            let _ = self.db.execute(
                "DELETE FROM metric_rollups WHERE instance_id=? AND host_id IS ?",
                &[&key, &host_id],
            );
        }
    }

    /// After a restart, count the series already in SQLite against their host's cap (once).
    fn adopt(&self) {
        let rows = self
            .db
            .all(
                "SELECT host_id, instance_id, MAX(bucket_ts) AS last FROM metric_rollups \
                 WHERE host_id IS NOT NULL GROUP BY host_id, instance_id",
                &[],
            )
            .unwrap_or_default();
        {
            let mut st = self.st();
            st.adopted = true;
            for r in rows {
                let host = r.str_or("host_id", "");
                let iid = r.str_or("instance_id", "");
                let last = r.f64("last").unwrap_or(0.0);
                let keys = st.keys.entry(host).or_default();
                if !keys.iter().any(|(k, _)| *k == iid) {
                    keys.push((iid, last));
                }
            }
        }
        let max = self.max_keys.load(Ordering::SeqCst);
        // an over-full host (cap lowered, or a past flood) is cut back now
        let hosts: Vec<String> = self.st().keys.keys().cloned().collect();
        for host in hosts {
            loop {
                let victim = {
                    let st = self.st();
                    let Some(keys) = st.keys.get(&host) else {
                        break;
                    };
                    if keys.len() <= max {
                        break;
                    }
                    let mut best: Option<(&String, f64)> = None;
                    for (k, last) in keys {
                        if best.is_none_or(|(_, b)| *last < b) {
                            best = Some((k, *last));
                        }
                    }
                    best.map(|(k, _)| k.clone())
                };
                match victim {
                    Some(v) => self.forget(&host, &v, true),
                    None => break,
                }
            }
        }
    }

    /// Drop series nobody updated for KEY_TTL_S, forget per-host bookkeeping of unknown hosts and delete rollup rows
    /// past their retention. Cheap and self-throttled: it runs at most once per PRUNE_EVERY_S.
    pub fn prune(&self, t: Option<f64>, force: bool) {
        let t = t.unwrap_or_else(|| self.now());
        {
            let mut st = self.st();
            if !force && t < st.next_prune {
                return;
            }
            st.next_prune = t + PRUNE_EVERY_S;
        }
        if !self.st().adopted {
            self.adopt();
        }
        let hosts: Vec<String> = self.st().keys.keys().cloned().collect();
        for host in hosts {
            let stale: Vec<String> = self
                .st()
                .keys
                .get(&host)
                .map(|keys| {
                    keys.iter()
                        .filter(|(_, last)| t - *last > KEY_TTL_S)
                        .map(|(k, _)| k.clone())
                        .collect()
                })
                .unwrap_or_default();
            for k in stale {
                self.forget(&host, &k, false); // the rows stay until their retention runs out
            }
            let mut st = self.st();
            if st.keys.get(&host).is_some_and(Vec::is_empty) {
                st.keys.remove(&host);
            }
        }
        let known: Vec<String> = self.fleet.hosts().into_iter().map(|h| h.id).collect();
        let since: HashMap<String, Option<f64>> = known
            .iter()
            .map(|h| {
                (
                    h.clone(),
                    self.fleet
                        .node(h)
                        .ok()
                        .and_then(|n| n.link().connected_since),
                )
            })
            .collect();
        {
            let mut st = self.st();
            st.last_seq.retain(|h, _| known.contains(h));
            st.configured.retain(|h, _| known.contains(h));
            // one warning per connection: older ones are dead weight
            st.warned
                .retain(|(h, cs)| since.get(h).is_some_and(|s| s == cs));
        }
        for (res, keep) in [("h", HOURLY_KEEP_S), ("d", DAILY_KEEP_S)] {
            #[allow(clippy::cast_possible_truncation)]
            let cutoff = (t - keep) as i64;
            let _ = self.db.execute(
                "DELETE FROM metric_rollups WHERE res=? AND bucket_ts < ?",
                &[&res, &cutoff],
            );
        }
    }

    /// Adapters report lifetime counters; derive windowed tok/s from consecutive scrapes when they didn't. Returns
    /// whether the prompt counter grew (the instance is in use).
    fn derive_rates(st: &mut State, iid: &str, t: f64, vals: &mut Vec<(String, f64)>) -> bool {
        let cur: Vec<(String, f64)> = ["prompt_tokens_total", "generation_tokens_total"]
            .iter()
            .filter_map(|k| get(vals, k).map(|v| ((*k).to_string(), v)))
            .collect();
        let mut touched = false;
        if let Some((pt, prev)) = st.last_counters.get(iid)
            && !cur.is_empty()
            && t > *pt
        {
            let dt = t - pt;
            for (total, rate) in [
                ("prompt_tokens_total", "prompt_tps"),
                ("generation_tokens_total", "generation_tps"),
            ] {
                if let (true, Some(c), Some(p)) = (
                    get(vals, rate).is_none(),
                    get(&cur, total),
                    get(prev, total),
                ) && c >= p
                {
                    put(vals, rate, (c - p) / dt);
                }
            }
            if get(&cur, "prompt_tokens_total").unwrap_or(0.0)
                > get(prev, "prompt_tokens_total").unwrap_or(0.0)
            {
                touched = true;
            }
        }
        st.last_counters.insert(iid.to_string(), (t, cur));
        touched
    }

    fn rollup(&self, key: &str, pt: &MetricPoint, host_id: &str) {
        let mut rows: Vec<Vec<SqlValue>> = Vec::new();
        for (res, step) in [("h", 3600.0), ("d", 86400.0)] {
            #[allow(clippy::cast_possible_truncation)]
            let bucket = ((pt.t / step).floor() * step) as i64;
            for (k, v) in &pt.values {
                let v = v.as_f64().unwrap_or(0.0);
                rows.push(vec![
                    SqlValue::Text(key.to_string()),
                    SqlValue::Text(res.into()),
                    SqlValue::Integer(bucket),
                    SqlValue::Text(k.clone()),
                    SqlValue::Real(v),
                    SqlValue::Integer(1),
                    SqlValue::Real(v),
                    SqlValue::Text(host_id.to_string()),
                ]);
            }
        }
        let _ = self.db.executemany(
            "INSERT INTO metric_rollups(instance_id,res,bucket_ts,key,sum,count,max,host_id) VALUES(?,?,?,?,?,?,?,?) \
             ON CONFLICT(instance_id,res,bucket_ts,key) DO UPDATE SET sum=sum+excluded.sum, count=count+1, \
             max=MAX(max, excluded.max)",
            &rows,
        );
    }

    // -- sampling loop --------------------------------------------------------------------------------------------------
    /// One round: system series per host, host RAM of every instance, and the pull fallback for managed engines.
    pub async fn scrape_once(&self) {
        let t = self.now();
        for h in self.fleet.hosts() {
            if self
                .fleet
                .node(&h.id)
                .is_ok_and(|n| ["up", "stale"].contains(&n.link().state.as_str()))
            {
                self.sample_system(&h.id, t);
            }
        }
        let mut pulls = Vec::new();
        for inst in self.lifecycle.all_instances() {
            if ["stopped", "failed", "unreachable"].contains(&inst.state.as_str())
                || !self.connected(&inst.host_id)
            {
                continue;
            }
            let (silent, since_pull) = {
                let st = self.st();
                (
                    t - st.last_push.get(&inst.id).copied().unwrap_or(0.0),
                    t - st.last_pull.get(&inst.id).copied().unwrap_or(0.0),
                )
            };
            if inst.management == "managed"
                && inst.state == "ready"
                && inst.port.is_some_and(|p| p != 0)
                && silent > 3.0 * self.interval_s
                && since_pull >= self.interval_s
            {
                self.st().last_pull.insert(inst.id.clone(), t);
                pulls.push(self.pull(inst, t));
            } else if silent > 2.0 * self.interval_s && inst.host_memory.is_some() {
                // no engine metrics: host RAM is still worth a series
                self.record(&inst, Vec::new(), t, false);
            }
        }
        futures::future::join_all(pulls).await;
    }

    fn connected(&self, host_id: &str) -> bool {
        self.fleet
            .node(host_id)
            .is_ok_and(|n| nv::link_up(n.as_ref()))
    }

    async fn pull(&self, inst: Instance, t: f64) {
        // a raise in the launch build fails this pull only (Python: gathered with return_exceptions=True)
        let Ok(launch) = self.lifecycle.launch_for_checked(&inst) else {
            return;
        };
        let path = launch.map_or_else(|| "/metrics".to_string(), |l| l.metrics_path);
        let opts = HttpOpts {
            timeout_s: Some(3.0),
            max_bytes: Some(2_000_000),
            ..HttpOpts::default()
        };
        let Ok(r) = self
            .http
            .request(&inst.host_id, inst.port.unwrap_or(0), "GET", &path, opts)
            .await
        else {
            return;
        };
        if r.status == 200 {
            let Ok(vals) = self.parse(&inst.engine, &String::from_utf8_lossy(&r.body)) else {
                return; // gathered with return_exceptions=True: a raising parser is swallowed
            };
            self.record(&inst, vals, t, false);
        }
    }

    fn sample_system(&self, host_id: &str, t: f64) {
        let Ok(node) = self.fleet.node(host_id) else {
            return;
        };
        let s = nv::sample(node.as_ref());
        let Some(m) = meminfo_from_sample(s.get("mem"), nv::num(s.get("t"))) else {
            return;
        };
        let seq = s.get("seq").cloned().unwrap_or(Value::Null);
        {
            let mut st = self.st();
            if !seq.is_null() && st.last_seq.get(host_id) == Some(&seq) {
                return; // nothing new from that node since the last round
            }
            st.last_seq.insert(host_id.to_string(), seq);
        }
        let key = system_key(host_id);
        let mut values = Map::new();
        values.insert("ram_used_gib".into(), json!(m.used_gib));
        values.insert("ram_available_gib".into(), json!(m.available_gib));
        values.insert("shmem_gib".into(), json!(m.shmem_gib));
        values.insert("swap_used_gib".into(), json!(m.swap_used_gib));
        let pt = MetricPoint { t, values };
        self.claim(host_id, &key, t);
        self.push_ring(&key, pt.clone());
        self.rollup(&key, &pt, host_id);
        let mut ev = Map::new();
        ev.insert("instance_id".into(), json!(SYSTEM_KEY));
        ev.insert("host_id".into(), json!(host_id));
        ev.extend(point_json(&pt));
        self.bus.publish(TOPIC, Value::Object(ev));
    }

    // -- queries --------------------------------------------------------------------------------------------------------
    /// GET /metrics/instances/{iid}: 404 for an unknown instance, 400 invalid_window.
    pub fn series(&self, iid: &str, window: &str) -> AppResult<Value> {
        let inst = self.lifecycle.get(iid)?;
        let mut extra = Map::new();
        if !inst.managed {
            extra.insert("history_since".into(), json!(inst.history_since));
        }
        self.series_of(iid, window, extra)
    }

    /// GET /metrics/system?host=: 404 unknown_host, 400 invalid_window.
    pub fn system_series(&self, window: &str, host_id: &str) -> AppResult<Value> {
        self.nodes.node(host_id)?;
        let mut v = self.series_of(&system_key(host_id), window, Map::new())?;
        if let Value::Object(o) = &mut v {
            o.insert("host_id".into(), json!(host_id));
        }
        Ok(v)
    }

    fn series_of(&self, key: &str, window: &str, extra: Map<String, Value>) -> AppResult<Value> {
        let secs = parse_window(window)?;
        #[allow(clippy::cast_precision_loss)]
        let since = self.now() - secs as f64;
        let mut out = Map::new();
        out.insert("instance_id".into(), json!(key));
        out.insert("window".into(), json!(window));
        if secs <= RING_SECONDS {
            let pts: Vec<Value> = self
                .st()
                .ring
                .get(key)
                .map(|r| {
                    r.iter()
                        .filter(|p| p.t >= since)
                        .map(|p| Value::Object(point_json(p)))
                        .collect()
                })
                .unwrap_or_default();
            out.insert("resolution".into(), json!("raw"));
            out.insert("points".into(), Value::Array(pts));
        } else {
            let (res, step) = if secs <= 3 * 86400 {
                ("h", 3600.0)
            } else {
                ("d", 86400.0)
            };
            #[allow(clippy::cast_possible_truncation)]
            let from = ((since / step).floor() * step) as i64;
            let rows = self.db.all(
                "SELECT bucket_ts, key, sum, count, max FROM metric_rollups WHERE instance_id=? AND res=? \
                 AND bucket_ts >= ? ORDER BY bucket_ts",
                &[&key, &res, &from],
            )?;
            let mut by_bucket: Vec<(i64, Map<String, Value>)> = Vec::new();
            for r in rows {
                let b = r.i64("bucket_ts").unwrap_or(0);
                let count = r.f64("count").unwrap_or(1.0);
                let avg = r.f64("sum").unwrap_or(0.0) / if count == 0.0 { 1.0 } else { count };
                let k = r.str_or("key", "");
                match by_bucket.iter_mut().find(|(bb, _)| *bb == b) {
                    Some((_, m)) => {
                        m.insert(k, json!(avg));
                    }
                    None => {
                        let mut m = Map::new();
                        m.insert(k, json!(avg));
                        by_bucket.push((b, m));
                    }
                }
            }
            by_bucket.sort_by_key(|(b, _)| *b);
            #[allow(clippy::cast_precision_loss)]
            let pts: Vec<Value> = by_bucket
                .into_iter()
                .map(|(b, v)| json!({"t": b as f64, "values": v}))
                .collect();
            out.insert(
                "resolution".into(),
                json!(if res == "h" { "1h" } else { "1d" }),
            );
            out.insert("points".into(), Value::Array(pts));
        }
        out.extend(extra);
        Ok(Value::Object(out))
    }

    pub fn latest(&self, iid: &str) -> Option<MetricPoint> {
        self.st().ring.get(iid).and_then(|r| r.back().cloned())
    }

    // -- running ------------------------------------------------------------------------------------------------------------
    /// Route every pushed `node.metrics` message (until cancelled).
    pub async fn consume(&self) {
        let mut sub = self.bus.subscribe(PUSH_TOPIC);
        while let Some(ev) = sub.recv().await {
            let host = ev
                .get("host_id")
                .map_or_else(|| "None".to_string(), ec_proto::pyfmt::str_value);
            match ev.get("msg") {
                Some(Value::Object(msg)) => {
                    // a message that makes ingest panic is dropped like a malformed one; the stream goes on
                    let _ =
                        crate::panics::guarded_sync("metrics ingest", || self.ingest(&host, msg));
                }
                // one malformed push must not stop the stream
                _ => {
                    tracing::error!(target: "engine_console.services.metrics", "could not process a pushed metrics message");
                }
            }
        }
    }

    async fn rounds(&self) {
        loop {
            // one failing round (a panic included) is logged and the next one runs (main.py `_loop`)
            crate::panics::guarded("metrics round", async {
                self.sync_hosts().await;
                self.scrape_once().await;
                self.prune(None, false);
            })
            .await;
            tokio::time::sleep(std::time::Duration::from_secs_f64(self.interval_s.max(0.0))).await;
        }
    }

    /// The background loop (configure hosts, scrape round, prune), until cancelled.
    pub async fn run(&self) {
        tokio::join!(self.consume(), self.rounds());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_window_accepts_digits_and_a_unit() {
        assert_eq!(parse_window("15m"), Ok(900));
        assert_eq!(parse_window("7d"), Ok(7 * 86400));
        for bad in ["", "m", "soon", "1w", "-1h", "1.5h", "²m"] {
            assert_eq!(
                parse_window(bad).map_err(|p| p.code),
                Err("invalid_window".to_string()),
                "{bad}"
            );
        }
        assert_eq!(system_key("local"), "system");
        assert_eq!(system_key("spark"), "system:spark");
    }
}
