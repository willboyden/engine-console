//! Python: node/src/ecnode/gpus.py.
//!
//! NVIDIA GPUs: NVML (libnvidia-ml.so.1, loaded at run time by nvml-wrapper; no link-time dependency) when it
//! initialises, else `nvidia-smi` CSV queries. Zero GPUs is fine. Unified-memory GPUs (e.g. GB10) report no memory
//! figures ("[N/A]" / NOT_SUPPORTED): they are reported with `unified_memory: true` and None for their memory and
//! per-process bytes, never a made-up number.
//!
//! `ECNODE_NO_NVML` (set and non-empty in the node's environment) skips NVML, as a Python node without the
//! `nvml` extra does (PORTING.md DV6; the parity harness sets it so both nodes read the same fake nvidia-smi).

use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard};

use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::error::NvmlError;
use serde_json::{Value, json};

use crate::util::{EnvMap, Runner, clean_text, py};

pub const MIB: f64 = 1024.0 * 1024.0;
pub const STATIC_FIELDS: &str =
    "index,uuid,name,memory.total,compute_cap,pci.bus_id,driver_version";
pub const STATIC_FIELDS_OLD: &str = "index,uuid,name,memory.total,pci.bus_id,driver_version";
pub const SAMPLE_FIELDS: &str = "uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed";
pub const PROC_FIELDS: &str = "pid,used_memory,gpu_uuid";
pub const NO_NVML_ENV: &str = "ECNODE_NO_NVML";

/// `float(s)` of one CSV cell, None for "[N/A]" and friends (and for non-finite or huge values).
fn num(s: &str) -> Option<f64> {
    let v = py::parse_float(s)?;
    (v.is_finite() && v.abs() < 1e15).then_some(v)
}

/// `csv.reader` rows of nvidia-smi output (cells stripped), blank rows skipped.
pub fn csv_rows(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let mut row: Vec<String> = Vec::new();
        let mut cell = String::new();
        let mut chars = line.chars().peekable();
        let mut quoted = false;
        let mut at_start = true;
        while let Some(c) = chars.next() {
            if quoted {
                if c == '"' {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        cell.push('"');
                    } else {
                        quoted = false;
                    }
                } else {
                    cell.push(c);
                }
            } else if c == '"' && at_start {
                quoted = true;
                at_start = false;
            } else if c == ',' {
                row.push(std::mem::take(&mut cell));
                at_start = true;
            } else {
                cell.push(c);
                at_start = false;
            }
        }
        if line.is_empty() {
            continue;
        }
        row.push(cell);
        let row: Vec<String> = row.iter().map(|c| py::strip(c).to_string()).collect();
        if row.iter().any(|c| !c.is_empty()) {
            rows.push(row);
        }
    }
    rows
}

/// `type(e).__name__` of pynvml's exception for an NVML error.
fn nvml_error_name(e: &NvmlError) -> &'static str {
    match e {
        NvmlError::Uninitialized => "NVMLError_Uninitialized",
        NvmlError::InvalidArg => "NVMLError_InvalidArgument",
        NvmlError::NotSupported => "NVMLError_NotSupported",
        NvmlError::NoPermission => "NVMLError_NoPermission",
        NvmlError::NotFound => "NVMLError_NotFound",
        NvmlError::InsufficientSize(_) => "NVMLError_InsufficientSize",
        NvmlError::DriverNotLoaded => "NVMLError_DriverNotLoaded",
        NvmlError::Timeout => "NVMLError_Timeout",
        NvmlError::GpuLost => "NVMLError_GpuIsLost",
        NvmlError::Unknown => "NVMLError_Unknown",
        _ => "NVMLError",
    }
}

struct NvmlState {
    nvml: Arc<Nvml>,
    /// uuid -> device index (Python keeps the handle; the index names the same device)
    handles: Vec<(String, u32)>,
}

struct Inner {
    nv: Option<NvmlState>,
    unified: HashSet<String>,
}

/// GPU_STATIC list and {driver, cuda, source, error}.
pub type Static = (Vec<Value>, Value);
/// ({uuid: GPU_SAMPLE without procs}, [(uuid, pid, used_bytes)])
pub type Sample = (
    Vec<(String, serde_json::Map<String, Value>)>,
    Vec<(String, i64, Option<i64>)>,
);

pub struct Gpus {
    smi: String,
    runner: Arc<dyn Runner>,
    inner: Mutex<Inner>,
}

impl Gpus {
    /// `Gpus(use_nvml=..., smi="nvidia-smi", runner=...)`
    pub fn new(use_nvml: bool, runner: Arc<dyn Runner>) -> Self {
        let nv = if use_nvml {
            Nvml::init().ok().map(|n| NvmlState {
                nvml: Arc::new(n),
                handles: Vec::new(),
            })
        } else {
            None
        };
        Self {
            smi: "nvidia-smi".into(),
            runner,
            inner: Mutex::new(Inner {
                nv,
                unified: HashSet::new(),
            }),
        }
    }

    /// NVML unless the node's environment says ECNODE_NO_NVML.
    pub fn for_env(env: &EnvMap, runner: Arc<dyn Runner>) -> Self {
        let off = env.get(NO_NVML_ENV).is_some_and(|v| !v.is_empty());
        Self::new(!off, runner)
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        crate::util::lock(&self.inner)
    }

    pub fn source(&self) -> &'static str {
        if self.lock().nv.is_some() {
            "nvml"
        } else {
            "nvidia-smi"
        }
    }

    // ── static ───────────────────────────────────────────────────────────────────────────────────────────
    pub fn static_info(&self) -> Static {
        let mut inner = self.lock();
        if inner.nv.is_some() {
            match static_nvml(&mut inner) {
                Ok(r) => return r,
                Err(e) => {
                    inner.nv = None;
                    let nvml_error =
                        format!("NVML failed ({}); using nvidia-smi", nvml_error_name(&e));
                    let (gpus, mut info) = self.static_smi(&mut inner);
                    if info["error"].is_null() {
                        info["error"] = json!(nvml_error);
                    }
                    return (gpus, info);
                }
            }
        }
        self.static_smi(&mut inner)
    }

    fn run_smi(&self, args: &[&str], timeout: f64) -> crate::util::CmdResult {
        let mut argv = vec![self.smi.clone()];
        argv.extend(args.iter().map(|s| (*s).to_string()));
        self.runner.run(&argv, timeout, None)
    }

    fn static_smi(&self, inner: &mut Inner) -> Static {
        let fmt = "--format=csv,noheader,nounits";
        let mut fields = STATIC_FIELDS;
        let mut res = self.run_smi(&[&format!("--query-gpu={fields}"), fmt], 8.0);
        if !res.ok() && format!("{}{}", res.out, res.err).contains("valid field") {
            fields = STATIC_FIELDS_OLD;
            res = self.run_smi(&[&format!("--query-gpu={fields}"), fmt], 8.0);
        }
        let none = |error: &str| {
            (
                Vec::new(),
                json!({"driver": null, "cuda": null, "source": "none", "error": error}),
            )
        };
        if res.rc == 127 {
            return none("no NVIDIA driver found (nvidia-smi is not on PATH)");
        }
        if !res.ok() {
            let text = if res.err.is_empty() {
                &res.out
            } else {
                &res.err
            };
            let text = py::strip(text);
            let msg = if text.is_empty() {
                format!("nvidia-smi exited with {}", res.rc)
            } else {
                text.to_string()
            };
            return none(&clean_text(&msg, 300));
        }
        let names: Vec<&str> = fields.split(',').collect();
        let mut out: Vec<Value> = Vec::new();
        let mut driver: Option<String> = None;
        inner.unified.clear();
        for row in csv_rows(&res.out) {
            if row.len() < names.len() {
                continue;
            }
            let get = |k: &str| names.iter().position(|n| *n == k).map(|i| row[i].as_str());
            let total = num(get("memory.total").unwrap_or(""));
            let uuid = get("uuid").unwrap_or("").to_string();
            if total.is_none() {
                inner.unified.insert(uuid.clone());
            }
            let idx = num(get("index").unwrap_or(""));
            let index = match idx {
                Some(i) => json!(py::int_of_float(i)),
                None => json!(out.len()),
            };
            let cc = get("compute_cap").filter(|c| num(c).is_some_and(|v| v != 0.0));
            let total_bytes = match total {
                Some(t) if t != 0.0 => json!(py::int_of_float(t * MIB)),
                _ => Value::Null,
            };
            out.push(json!({
                "index": index, "uuid": uuid, "name": clean_text(get("name").unwrap_or(""), 128),
                "total_bytes": total_bytes, "compute_capability": cc, "unified_memory": total.is_none(),
                "pci_bus_id": get("pci.bus_id").filter(|p| !p.is_empty()),
            }));
            if driver.is_none() {
                driver = get("driver_version")
                    .filter(|d| !d.is_empty())
                    .map(str::to_string);
            }
        }
        let mut cuda: Option<String> = None;
        if !out.is_empty() {
            let head = self.run_smi(&[], 8.0);
            if head.ok() {
                cuda = cuda_version(&head.out);
            }
        }
        (
            out,
            json!({"driver": driver, "cuda": cuda, "source": "nvidia-smi", "error": null}),
        )
    }

    // ── live ─────────────────────────────────────────────────────────────────────────────────────────────
    pub fn sample(&self) -> Sample {
        let inner = self.lock();
        if let Some(nv) = &inner.nv
            && !nv.handles.is_empty()
        {
            return sample_nvml(nv, &inner.unified).unwrap_or_default();
        }
        drop(inner);
        self.sample_smi()
    }

    fn sample_smi(&self) -> Sample {
        let fmt = "--format=csv,noheader,nounits";
        let res = self.run_smi(&[&format!("--query-gpu={SAMPLE_FIELDS}"), fmt], 10.0);
        let mut gpus: Vec<(String, serde_json::Map<String, Value>)> = Vec::new();
        if !res.ok() {
            return (gpus, Vec::new());
        }
        for row in csv_rows(&res.out) {
            if row.len() < 8 {
                continue;
            }
            let (used, total) = (num(&row[2]), num(&row[3]));
            let m = json!({
                "util_pct": num(&row[1]),
                "mem_used_bytes": used.map(|u| py::int_of_float(u * MIB)),
                "mem_total_bytes": total.map(|t| py::int_of_float(t * MIB)),
                "temp_c": num(&row[4]), "power_w": num(&row[5]), "power_limit_w": num(&row[6]),
                "fan_pct": num(&row[7]),
            });
            let Value::Object(m) = m else { continue };
            let uuid = row[0].clone();
            gpus.retain(|(u, _)| *u != uuid);
            gpus.push((uuid, m));
        }
        let mut procs = Vec::new();
        let pres = self.run_smi(&[&format!("--query-compute-apps={PROC_FIELDS}"), fmt], 10.0);
        if pres.ok() {
            for row in csv_rows(&pres.out) {
                if row.len() < 3 || !py::isdigit(&row[0]) {
                    continue;
                }
                let Ok(pid) = row[0].parse::<i64>() else {
                    continue;
                };
                let used = num(&row[1]).map(|u| py::int_of_float(u * MIB));
                procs.push((row[2].clone(), pid, used));
            }
        }
        (gpus, procs)
    }
}

fn cuda_version(text: &str) -> Option<String> {
    static CUDA: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)")
            .unwrap_or_else(|_| std::process::abort())
    });
    CUDA.captures(text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

fn static_nvml(inner: &mut Inner) -> Result<Static, NvmlError> {
    let Some(nv) = inner.nv.as_mut() else {
        return Err(NvmlError::Uninitialized);
    };
    let nvml = Arc::clone(&nv.nvml);
    let mut out = Vec::new();
    nv.handles.clear();
    inner.unified.clear();
    for i in 0..nvml.device_count()? {
        let h = nvml.device_by_index(i)?;
        let uuid = h.uuid()?;
        let total = match h.memory_info() {
            Ok(m) => Some(i64::try_from(m.total).unwrap_or(i64::MAX)),
            Err(_) => {
                inner.unified.insert(uuid.clone());
                None
            }
        };
        let cc = h
            .cuda_compute_capability()
            .ok()
            .map(|c| format!("{}.{}", c.major, c.minor));
        let pci = h.pci_info().ok().map(|p| p.bus_id);
        let name = h.name()?;
        nv.handles.push((uuid.clone(), i));
        out.push(json!({
            "index": i, "uuid": uuid, "name": clean_text(&name, 128), "total_bytes": total,
            "compute_capability": cc, "unified_memory": total.is_none(), "pci_bus_id": pci,
        }));
    }
    let (mut driver, mut cuda) = (None, None);
    if let Ok(d) = nvml.sys_driver_version() {
        driver = Some(d);
        if let Ok(v) = nvml.sys_cuda_driver_version() {
            cuda = Some(format!("{}.{}", v / 1000, (v % 1000) / 10));
        }
    }
    Ok((
        out,
        json!({"driver": driver, "cuda": cuda, "source": "nvml", "error": null}),
    ))
}

fn sample_nvml(nv: &NvmlState, unified: &HashSet<String>) -> Option<Sample> {
    let mut gpus = Vec::new();
    let mut procs = Vec::new();
    for (uuid, idx) in &nv.handles {
        let h = nv.nvml.device_by_index(*idx).ok()?;
        let mem = if unified.contains(uuid) {
            None
        } else {
            h.memory_info().ok()
        };
        let util = h.utilization_rates().ok();
        let power = h.power_usage().ok();
        let limit = h.enforced_power_limit().ok();
        let m = json!({
            "util_pct": util.map(|u| f64::from(u.gpu)),
            "mem_used_bytes": mem.as_ref().map(|m| m.used),
            "mem_total_bytes": mem.as_ref().map(|m| m.total),
            "temp_c": h.temperature(TemperatureSensor::Gpu).ok().map(f64::from),
            "power_w": power.map(|p| f64::from(p) / 1000.0),
            "power_limit_w": limit.map(|p| f64::from(p) / 1000.0),
            "fan_pct": h.fan_speed(0).ok().map(f64::from),
        });
        if let Value::Object(m) = m {
            gpus.push((uuid.clone(), m));
        }
        for p in h.running_compute_processes().unwrap_or_default() {
            let used = match p.used_gpu_memory {
                UsedGpuMemory::Used(b) if !unified.contains(uuid) => i64::try_from(b).ok(),
                _ => None,
            };
            procs.push((uuid.clone(), i64::from(p.pid), used));
        }
    }
    Some((gpus, procs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_rows_strip_cells_and_skip_blank_rows() {
        // as Python's csv.reader: a quote opens a quoted field only at the very start of the field
        assert_eq!(
            csv_rows("0, GPU-a, \"NVIDIA, Inc\", 97887\n\n , \n1, GPU-b, x, [N/A]\n\"q,1\",2\n"),
            vec![
                vec!["0", "GPU-a", "\"NVIDIA", "Inc\"", "97887"],
                vec!["1", "GPU-b", "x", "[N/A]"],
                vec!["q,1", "2"]
            ]
        );
    }
}
