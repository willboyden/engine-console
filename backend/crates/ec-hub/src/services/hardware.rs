//! Python: backend/src/engine_console/services/hardware.py (tests/core/test_hardware.py).
//!
//! Per-host hardware view, built from what each host's node pushed (hello = static facts, sample = live numbers).
//!
//! The hub never probes NVML or /proc itself (contract v2). `report(host_id)` is the API shape (v1 plus `host_id`);
//! `hardware(host_id, gpu_ids)` is the adapter-facing view the fit estimator and the launch builders use.
//!
//! Unified-memory machines (a GPU whose memory IS the host's RAM, e.g. GB10) cannot say how much VRAM is used, so
//! their GPU budget is taken from the host RAM numbers: one pool, counted once.

use std::sync::Arc;

use serde_json::Value;

use crate::adapters::base::Hardware;
use crate::domain::errors::{AppResult, NodeError, Obj};
use crate::domain::fleet::{FleetPort, NodeClient};
use crate::domain::models::{GpuStat, HardwareReport, HostMemInfo};
use crate::services::hostmem::{alerts_for, meminfo_from_sample};
use crate::services::nodeview as nv;

/// (node, hello, sample) of a host whose link is live
type Required = (Arc<dyn NodeClient>, Arc<Obj>, Arc<Obj>);

pub struct HardwareService {
    fleet: Arc<dyn FleetPort>,
}

impl std::fmt::Debug for HardwareService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HardwareService")
    }
}

fn truthy(v: Option<&Value>) -> bool {
    v.is_some_and(ec_proto::sanitize::truthy)
}

/// Python's `int(x)` of a GPU index (truncation; indexes are small).
#[allow(clippy::cast_possible_truncation)]
fn f64_to_i64(x: f64) -> i64 {
    x as i64
}

fn mem_of(sample: &Obj) -> Option<HostMemInfo> {
    meminfo_from_sample(sample.get("mem"), nv::num(sample.get("t")))
}

impl HardwareService {
    pub fn new(fleet: Arc<dyn FleetPort>) -> Self {
        Self { fleet }
    }

    /// 404 unknown_host for an unconfigured host.
    pub fn node(&self, host_id: &str) -> AppResult<Arc<dyn NodeClient>> {
        Ok(self.fleet.node(host_id)?)
    }

    /// (node, hello, sample) or `node_unavailable`: numbers from a link that is gone would be stale, not live.
    fn require(&self, host_id: &str) -> AppResult<Required> {
        let node = self.node(host_id)?;
        let link = node.link();
        let hello = node.latest("hello").filter(|h| !h.is_empty());
        let live = link.state == "up" || link.state == "stale";
        let Some(hello) = hello.filter(|_| live) else {
            let why = link
                .error
                .clone()
                .unwrap_or_else(|| format!("the node is {}: no hardware data", link.state));
            return Err(NodeError::unavailable(host_id, why).into());
        };
        let Some(sample) = node.latest("sample") else {
            return Err(NodeError::unavailable(
                host_id,
                "the node has not sent its first sample yet; try again in a few seconds",
            )
            .into());
        };
        Ok((node, hello, sample))
    }

    /// Host RAM of a connected host (None when its link is not up or stale, or it sent no memory).
    pub fn host_memory(&self, host_id: &str) -> AppResult<Option<HostMemInfo>> {
        let node = self.node(host_id)?;
        let state = node.link().state;
        if state != "up" && state != "stale" {
            return Ok(None);
        }
        Ok(mem_of(&nv::sample(node.as_ref())))
    }

    /// GPUs of a host from its last sample (node_unavailable when its link is down or no sample came yet).
    pub fn gpus(&self, host_id: &str) -> AppResult<Vec<GpuStat>> {
        let (node, _hello, sample) = self.require(host_id)?;
        let mem = mem_of(&sample);
        let live: Obj = match sample.get("gpus") {
            Some(Value::Object(m)) => m.clone(),
            _ => Obj::new(),
        };
        let mut gpus = nv::static_gpus(node.as_ref());
        gpus.sort_by(|a, b| {
            let k = |g: &Obj| nv::num(g.get("index")).filter(|x| *x != 0.0).unwrap_or(0.0);
            k(a).total_cmp(&k(b))
        });
        let mut out = Vec::new();
        for g in gpus {
            let uuid = g.get("uuid").map(nv::value_str).unwrap_or_default();
            let s: Obj = match live.get(&uuid) {
                Some(Value::Object(o)) => o.clone(),
                _ => Obj::new(),
            };
            let mut total = nv::gib(g.get("total_bytes"));
            let mut used = nv::gib(s.get("mem_used_bytes"));
            if truthy(g.get("unified_memory"))
                && used.is_none()
                && let Some(m) = &mem
            {
                // the pool is the host's RAM
                let t = total.filter(|t| *t != 0.0).unwrap_or(m.total_gib);
                total = Some(t);
                used = Some((t - m.available_gib).max(0.0));
            }
            let total = total.unwrap_or_else(|| {
                nv::gib(s.get("mem_total_bytes"))
                    .filter(|t| *t != 0.0)
                    .unwrap_or(0.0)
            });
            let used = used.unwrap_or(0.0);
            out.push(GpuStat {
                index: nv::num(g.get("index"))
                    .filter(|x| *x != 0.0)
                    .map_or(0, f64_to_i64),
                uuid,
                name: match g.get("name") {
                    Some(v) if ec_proto::sanitize::truthy(v) => nv::value_str(v),
                    _ => "GPU".into(),
                },
                total_gib: total,
                free_gib: (total - used).max(0.0),
                used_gib: used,
                util_pct: nv::num(s.get("util_pct")),
                temp_c: nv::num(s.get("temp_c")),
                power_w: nv::num(s.get("power_w")),
                fan_pct: nv::num(s.get("fan_pct")),
                compute_capability: g
                    .get("compute_capability")
                    .and_then(Value::as_str)
                    .filter(|c| !c.is_empty())
                    .map(str::to_string),
            });
        }
        Ok(out)
    }

    fn ram(hello: &Obj) -> Option<&Value> {
        hello.get("host").and_then(|h| h.get("ram_total_bytes"))
    }

    /// GET /hardware?host=
    pub fn report(&self, host_id: &str) -> AppResult<HardwareReport> {
        let (_node, hello, sample) = self.require(host_id)?;
        let mem = mem_of(&sample);
        let source = hello
            .get("gpu")
            .and_then(|g| g.get("source"))
            .filter(|v| ec_proto::sanitize::truthy(v))
            .map_or_else(|| "none".to_string(), nv::value_str);
        Ok(HardwareReport {
            gpus: self.gpus(host_id)?,
            host_ram_gib: mem.as_ref().map_or_else(
                || {
                    nv::gib(Self::ram(&hello))
                        .filter(|x| *x != 0.0)
                        .unwrap_or(0.0)
                },
                |m| m.total_gib,
            ),
            host_ram_free_gib: mem.as_ref().map_or(0.0, |m| m.available_gib),
            source,
            alerts: mem.as_ref().map(alerts_for).unwrap_or_default(),
            memory: mem,
            host_id: host_id.to_string(),
        })
    }

    /// The adapter-facing view restricted to `gpu_ids` (None or empty = all GPUs of that host).
    pub fn hardware(&self, host_id: &str, gpu_ids: Option<&[i64]>) -> AppResult<Hardware> {
        let (node, hello, sample) = self.require(host_id)?;
        let rep = self.gpus(host_id)?;
        let sel: Vec<GpuStat> = rep
            .into_iter()
            .filter(|g| gpu_ids.is_none_or(|ids| ids.is_empty() || ids.contains(&g.index)))
            .collect();
        let unified: Vec<(String, bool)> = nv::static_gpus(node.as_ref())
            .iter()
            .map(|g| {
                (
                    g.get("uuid").map(nv::value_str).unwrap_or_default(),
                    truthy(g.get("unified_memory")),
                )
            })
            .collect();
        let mut ccs: Vec<String> = sel
            .iter()
            .filter_map(|g| g.compute_capability.clone())
            .collect();
        ccs.sort();
        ccs.dedup();
        let mem = mem_of(&sample);
        let ram = mem.as_ref().map_or_else(
            || {
                nv::gib(Self::ram(&hello))
                    .filter(|x| *x != 0.0)
                    .unwrap_or(0.0)
            },
            |m| m.total_gib,
        );
        let is_unified = |u: &str| unified.iter().any(|(k, v)| k == u && *v);
        Ok(Hardware {
            gpu_ids: sel.iter().map(|g| g.index).collect(),
            gpu_uuids: sel.iter().map(|g| g.uuid.clone()).collect(),
            gpu_names: sel.iter().map(|g| g.name.clone()).collect(),
            gpu_total_gib: sel.iter().map(|g| g.total_gib).collect(),
            gpu_free_gib: sel.iter().map(|g| g.free_gib).collect(),
            compute_capability: ccs.last().cloned().unwrap_or_else(|| "unknown".into()),
            host_ram_gib: ram,
            host_id: host_id.to_string(),
            arch: nv::arch(node.as_ref()),
            unified_memory: sel.iter().any(|g| is_unified(&g.uuid)),
        })
    }
}
