//! Host RAM accounting: pure functions (port of services/hostmem.py).
//! The hub never reads /proc or cgroups; nodes push `sample` messages and this turns them into API shapes.

use serde_json::Value;

use crate::domain::errors::Obj;
use crate::domain::models::{HostAlert, HostMemInfo, HostMemory};

pub const GIB: f64 = 1_073_741_824.0;

fn kb_to_gib(kb: f64) -> f64 {
    kb * 1024.0 / GIB
}

/// A finite non-negative number from untrusted node data, else None (bool is not a number here).
pub fn num(v: Option<&Value>) -> Option<f64> {
    match v {
        Some(Value::Number(n)) => n.as_f64().filter(|x| *x >= 0.0 && x.is_finite()),
        _ => None,
    }
}

pub fn parse_meminfo(text: &str, now: f64) -> Option<HostMemInfo> {
    let mut vals: Vec<(String, f64)> = Vec::new();
    for line in crate::services::common::py_splitlines(text) {
        let (key, rest) = line.split_once(':').unwrap_or((line, ""));
        if let Some(first) = rest.split_whitespace().next()
            && let Ok(x) = first.parse::<f64>()
        {
            if let Some(slot) = vals.iter_mut().find(|(k, _)| k == key) {
                slot.1 = x;
            } else {
                vals.push((key.to_string(), x));
            }
        }
    }
    let get = |k: &str| vals.iter().find(|(kk, _)| kk == k).map(|(_, v)| *v);
    let (total, avail) = (get("MemTotal")?, get("MemAvailable")?);
    let swap_total = get("SwapTotal").unwrap_or(0.0);
    let swap_free = get("SwapFree").unwrap_or(0.0);
    Some(HostMemInfo {
        total_gib: kb_to_gib(total),
        used_gib: kb_to_gib(total - avail),
        available_gib: kb_to_gib(avail),
        free_gib: kb_to_gib(get("MemFree").unwrap_or(0.0)),
        cached_gib: kb_to_gib(get("Cached").unwrap_or(0.0)),
        shmem_gib: kb_to_gib(get("Shmem").unwrap_or(0.0)),
        swap_total_gib: kb_to_gib(swap_total),
        swap_used_gib: kb_to_gib((swap_total - swap_free).max(0.0)),
        updated_at: now,
    })
}

/// HostMemInfo from the `mem` block of a node `sample` (bytes). None when the node could not read it.
pub fn meminfo_from_sample(mem: Option<&Value>, t: Option<f64>) -> Option<HostMemInfo> {
    let mem: &Obj = match mem {
        Some(Value::Object(m)) if !m.is_empty() => m,
        _ => return None,
    };
    let (total, avail) = (num(mem.get("total"))?, num(mem.get("available"))?);
    let swap_total = num(mem.get("swap_total")).unwrap_or(0.0);
    let swap_free = num(mem.get("swap_free")).unwrap_or(0.0);
    Some(HostMemInfo {
        total_gib: total / GIB,
        used_gib: (total - avail).max(0.0) / GIB,
        available_gib: avail / GIB,
        free_gib: num(mem.get("free")).unwrap_or(0.0) / GIB,
        cached_gib: num(mem.get("cached")).unwrap_or(0.0) / GIB,
        shmem_gib: num(mem.get("shmem")).unwrap_or(0.0) / GIB,
        swap_total_gib: swap_total / GIB,
        swap_used_gib: (swap_total - swap_free).max(0.0) / GIB,
        updated_at: t.unwrap_or(0.0),
    })
}

/// HostMemory of one container from the node's cgroup block (cache = file - shmem).
pub fn container_memory(cg: Option<&Value>) -> Option<HostMemory> {
    let cg = match cg {
        Some(Value::Object(m)) if !m.is_empty() => m,
        _ => return None,
    };
    let current = num(cg.get("current"))?;
    let shmem = num(cg.get("shmem")).unwrap_or(0.0);
    let file = num(cg.get("file")).unwrap_or(0.0);
    let source = if cg.get("source").and_then(Value::as_str) == Some("rss") {
        "rss"
    } else {
        "cgroup"
    };
    Some(HostMemory {
        total_gib: current / GIB,
        anon_gib: num(cg.get("anon")).unwrap_or(0.0) / GIB,
        cache_gib: (file - shmem).max(0.0) / GIB,
        shmem_gib: shmem / GIB,
        kernel_gib: num(cg.get("kernel")).unwrap_or(0.0) / GIB,
        source: source.into(),
    })
}

/// One figure for an engine made of several containers.
pub fn sum_memory(parts: &[HostMemory]) -> Option<HostMemory> {
    if parts.is_empty() {
        return None;
    }
    Some(HostMemory {
        total_gib: parts.iter().map(|p| p.total_gib).sum(),
        anon_gib: parts.iter().map(|p| p.anon_gib).sum(),
        cache_gib: parts.iter().map(|p| p.cache_gib).sum(),
        shmem_gib: parts.iter().map(|p| p.shmem_gib).sum(),
        kernel_gib: parts.iter().map(|p| p.kernel_gib).sum(),
        source: if parts.iter().any(|p| p.source == "rss") {
            "rss"
        } else {
            "cgroup"
        }
        .into(),
    })
}

/// Every code `alerts_for` can raise (the learn gate requires an `alert.<code>` explanation for each; a unit test
/// keeps this list equal to what the function emits).
pub const ALERT_CODES: &[&str] = &["host_ram_low", "shmem_high", "swap_heavy"];

pub fn alerts_for(m: &HostMemInfo) -> Vec<HostAlert> {
    let mut out = Vec::new();
    let alert = |level: &str, code: &str, message: String| HostAlert {
        level: level.into(),
        code: code.into(),
        message,
    };
    if m.total_gib > 0.0 {
        let frac = m.available_gib / m.total_gib;
        if frac < 0.05 {
            out.push(alert(
                "crit",
                "host_ram_low",
                format!(
                    "Only {:.1} GiB of {:.0} GiB host RAM is available ({:.0}%): new engines may be OOM-killed.",
                    m.available_gib,
                    m.total_gib,
                    frac * 100.0
                ),
            ));
        } else if frac < 0.10 {
            out.push(alert(
                "warn",
                "host_ram_low",
                format!(
                    "Host RAM is getting low: {:.1} GiB of {:.0} GiB available ({:.0}%).",
                    m.available_gib,
                    m.total_gib,
                    frac * 100.0
                ),
            ));
        }
        if m.shmem_gib > 0.25 * m.total_gib {
            out.push(alert(
                "warn",
                "shmem_high",
                format!(
                    "{:.1} GiB of host RAM is shared memory. It counts as page cache but cannot be reclaimed under \
                     pressure, so it is not really available.",
                    m.shmem_gib
                ),
            ));
        }
    }
    if m.swap_total_gib > 0.0 && m.swap_used_gib > 0.5 * m.swap_total_gib {
        out.push(alert(
            "warn",
            "swap_heavy",
            format!(
                "Swap is {:.0}% used ({:.1} of {:.0} GiB): the host is under memory pressure.",
                m.swap_used_gib / m.swap_total_gib * 100.0,
                m.swap_used_gib,
                m.swap_total_gib
            ),
        ));
    }
    out
}

#[cfg(test)]
mod alert_code_tests {
    use super::*;

    fn info(available: f64, shmem: f64, swap_used: f64) -> HostMemInfo {
        HostMemInfo {
            total_gib: 100.0,
            used_gib: 100.0 - available,
            available_gib: available,
            free_gib: available,
            cached_gib: 0.0,
            shmem_gib: shmem,
            swap_total_gib: 10.0,
            swap_used_gib: swap_used,
            updated_at: 1.0,
        }
    }

    #[test]
    fn test_alert_codes_list_exactly_what_alerts_for_emits() {
        let mut seen: Vec<String> = [
            info(1.0, 50.0, 9.0),
            info(7.0, 0.0, 0.0),
            info(90.0, 0.0, 0.0),
        ]
        .iter()
        .flat_map(alerts_for)
        .map(|a| a.code)
        .collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen, ALERT_CODES);
    }
}
