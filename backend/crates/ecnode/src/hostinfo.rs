//! Python: node/src/ecnode/hostinfo.py.
//!
//! Static facts about this machine for `hello`: OS, kernel, CPU, RAM, boot time, node id, interface addresses.
//! Filesystem roots are injectable (`Paths`) so tests build fake /proc, /sys and /etc trees.

use std::net::IpAddr;
use std::path::{Path, PathBuf};

use ec_proto::Obj;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::util::{Runner, clean_text, py};

/// The machine trees the node reads (tests point them at temp dirs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    pub proc: PathBuf,
    pub sys: PathBuf,
    pub etc: PathBuf,
    /// ~/.local/state/ecnode: the generated engine config files live below it
    pub state: PathBuf,
}

impl Default for Paths {
    fn default() -> Self {
        Self {
            proc: PathBuf::from("/proc"),
            sys: PathBuf::from("/sys"),
            etc: PathBuf::from("/etc"),
            state: default_state(),
        }
    }
}

/// `~/.local/state/ecnode`
pub fn default_state() -> PathBuf {
    PathBuf::from(crate::policy::home_dir())
        .join(".local")
        .join("state")
        .join("ecnode")
}

impl Paths {
    pub fn cgroup(&self) -> PathBuf {
        self.sys.join("fs").join("cgroup")
    }
}

/// `open(path, "rb").read(limit).decode("utf-8", "replace")`, None on any OSError. Every /proc, /sys and /etc read
/// of the node goes through here (memory.rs and disks.rs too): symlinks are followed as Python's `open` does
/// (/etc/os-release usually is one), but the open never blocks on a FIFO, anything that is not a regular file
/// (FIFO, socket, device, directory) is None where Python would hang on a FIFO, and at most `limit` bytes are read
/// whatever size the file reports (pseudo-files report 0). (`util::fsread`, node hardening)
pub fn read_text(path: &Path, limit: usize) -> Option<String> {
    let buf = crate::util::fsread::read_file(path, limit, true).ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

fn hex16(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))[..16].to_string()
}

/// `socket.gethostname()`
pub fn hostname() -> String {
    nix::unistd::gethostname()
        .map(|h| h.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// First 16 hex of sha256(/etc/machine-id): stable per machine, reveals nothing about it.
pub fn node_id(paths: &Paths) -> String {
    let parent = py::dirname(&paths.etc);
    for p in [
        paths.etc.join("machine-id"),
        parent
            .join("var")
            .join("lib")
            .join("dbus")
            .join("machine-id"),
    ] {
        let text = read_text(&p, 256).unwrap_or_default();
        let text = py::strip(&text);
        if !text.is_empty() {
            return hex16(text.as_bytes());
        }
    }
    hex16(format!("host:{}", hostname()).as_bytes())
}

pub fn os_release(paths: &Paths) -> Option<String> {
    let parent = py::dirname(&paths.etc);
    for p in [
        paths.etc.join("os-release"),
        parent.join("usr").join("lib").join("os-release"),
    ] {
        let Some(text) = read_text(&p, 65536).filter(|t| !t.is_empty()) else {
            continue;
        };
        let mut vals: Vec<(String, String)> = Vec::new();
        for line in py::splitlines(&text) {
            if let Some((k, v)) = line.split_once('=') {
                let v = py::strip_chars(py::strip_chars(py::strip(v), "\""), "'");
                let k = py::strip(k).to_string();
                vals.retain(|(x, _)| *x != k);
                vals.push((k, v.to_string()));
            }
        }
        let get = |k: &str| {
            vals.iter()
                .find(|(x, _)| x == k)
                .map(|(_, v)| v.as_str())
                .filter(|v| !v.is_empty())
        };
        let name = get("PRETTY_NAME").map(str::to_string).unwrap_or_else(|| {
            [get("NAME"), get("VERSION")]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" ")
        });
        if !name.is_empty() {
            return Some(clean_text(&name, 128));
        }
    }
    None
}

/// `os.cpu_count()`
#[allow(clippy::useless_conversion)] // c_long is i64 on the 64-bit targets, not on every target
pub fn cpu_count() -> Option<i64> {
    nix::unistd::sysconf(nix::unistd::SysconfVar::_NPROCESSORS_ONLN)
        .ok()
        .flatten()
        .map(i64::from)
        .filter(|n| *n > 0)
}

/// `platform.processor()` on Linux: `uname -p`, "unknown" read as blank.
fn processor(runner: &dyn Runner) -> String {
    let res = runner.run(&["uname".to_string(), "-p".to_string()], 5.0, None);
    if !res.ok() {
        return String::new();
    }
    let p = py::strip(&res.out).to_string();
    if p == "unknown" { String::new() } else { p }
}

/// (model, physical cores, threads)
pub fn cpu_facts(paths: &Paths, runner: &dyn Runner) -> (Option<String>, Option<i64>, Option<i64>) {
    let text = read_text(&paths.proc.join("cpuinfo"), 4 << 20).unwrap_or_default();
    let mut model: Option<String> = None;
    let mut cores: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    let mut phys = "0".to_string();
    for line in py::splitlines(&text) {
        let (k, v) = line.split_once(':').unwrap_or((line, ""));
        let (k, v) = (py::strip(k), py::strip(v));
        if (k == "model name" || k == "Hardware") && model.is_none() && !v.is_empty() {
            model = Some(v.to_string());
        } else if k == "physical id" {
            phys = v.to_string();
        } else if k == "core id" {
            cores.insert((phys.clone(), v.to_string()));
        }
    }
    if model.is_none() {
        let dt = read_text(
            &paths
                .sys
                .join("firmware")
                .join("devicetree")
                .join("base")
                .join("model"),
            256,
        );
        model = match dt.filter(|d| !d.is_empty()) {
            Some(d) => Some(py::strip_chars(&d, "\u{0} \n").to_string()),
            None => Some(processor(runner)).filter(|p| !p.is_empty()),
        };
    }
    let threads = cpu_count();
    let n_cores = if cores.is_empty() {
        threads
    } else {
        i64::try_from(cores.len()).ok()
    };
    (
        model.filter(|m| !m.is_empty()).map(|m| clean_text(&m, 128)),
        n_cores,
        threads,
    )
}

pub fn meminfo_total(paths: &Paths) -> Option<i64> {
    let text = read_text(&paths.proc.join("meminfo"), 65536).unwrap_or_default();
    for line in py::splitlines(&text) {
        if line.starts_with("MemTotal:") {
            let parts = py::split_ws(line);
            if parts.len() >= 2 && py::isdigit(parts[1]) {
                return parts[1].parse::<i64>().ok().map(|k| k * 1024);
            }
        }
    }
    None
}

pub fn boot_time(paths: &Paths) -> Option<f64> {
    let text = read_text(&paths.proc.join("stat"), 1 << 20).unwrap_or_default();
    for line in py::splitlines(&text) {
        if line.starts_with("btime ") {
            let v = py::split_ws(line).get(1).copied().unwrap_or("");
            return if py::isdigit(v) {
                v.parse::<f64>().ok()
            } else {
                None
            };
        }
    }
    None
}

/// `platform.machine()` and `platform.release()`
fn uname() -> (String, String) {
    match nix::sys::utsname::uname() {
        Ok(u) => (
            u.machine().to_string_lossy().into_owned(),
            u.release().to_string_lossy().into_owned(),
        ),
        Err(_) => (std::env::consts::ARCH.to_string(), String::new()),
    }
}

/// hello.host (`python` is null: there is no Python in this node, PORTING.md DV5).
pub fn host_facts(paths: &Paths, runner: &dyn Runner) -> Obj {
    let (model, cores, threads) = cpu_facts(paths, runner);
    let (machine, release) = uname();
    let v = json!({
        "hostname": clean_text(&hostname(), 253), "arch": machine, "os": os_release(paths),
        "kernel": clean_text(&release, 128), "cpu_model": model, "cpu_cores": cores, "cpu_threads": threads,
        "ram_total_bytes": meminfo_total(paths), "boot_time": boot_time(paths), "python": null,
    });
    match v {
        Value::Object(o) => o,
        _ => Obj::new(),
    }
}

const SKIP_IFACES: &[&str] = &[
    "lo", "veth", "docker", "br-", "virbr", "cni", "flannel", "tap", "tun",
];

fn skipped(name: &str) -> bool {
    SKIP_IFACES.iter().any(|p| name.starts_with(p))
}

/// Not loopback, link-local, multicast or unspecified (`ipaddress` properties).
pub fn usable(addr: &str) -> bool {
    match addr.parse::<IpAddr>() {
        Ok(IpAddr::V4(a)) => {
            !(a.is_loopback() || a.is_link_local() || a.is_multicast() || a.is_unspecified())
        }
        Ok(IpAddr::V6(a)) => {
            let link_local = a.segments()[0] & 0xffc0 == 0xfe80;
            !(a.is_loopback() || link_local || a.is_multicast() || a.is_unspecified())
        }
        Err(_) => false,
    }
}

/// IPv4/IPv6 addresses of real interfaces (no loopback, link-local or container bridges).
pub fn addresses(runner: &dyn Runner, paths: &Paths) -> Vec<Value> {
    let res = runner.run(&["ip".into(), "-j".into(), "addr".into()], 3.0, None);
    if !res.ok() {
        let mut out = fallback_addresses(paths);
        out.truncate(64);
        return out;
    }
    let data: Value = serde_json::from_str(&res.out).unwrap_or(Value::Array(Vec::new()));
    let mut out = Vec::new();
    for iface in data.as_array().map(Vec::as_slice).unwrap_or_default() {
        let Some(name) = iface.get("ifname").and_then(Value::as_str) else {
            continue;
        };
        if skipped(name) {
            continue;
        }
        let infos = iface.get("addr_info").and_then(Value::as_array);
        for a in infos.map(Vec::as_slice).unwrap_or_default() {
            let Value::Object(a) = a else { continue };
            let (Some(Value::String(addr)), Some(prefix)) = (a.get("local"), a.get("prefixlen"))
            else {
                continue;
            };
            let prefix = match prefix {
                Value::Number(n) if !n.is_f64() => prefix.clone(),
                Value::Bool(b) => json!(i64::from(*b)),
                _ => continue,
            };
            if usable(addr) {
                out.push(json!({"iface": clean_text(name, 32), "addr": addr, "prefix": prefix}));
            }
        }
    }
    out.truncate(64);
    out
}

fn hex_le_ipv4(h: &str) -> Option<u32> {
    u32::from_str_radix(h, 16).ok().map(u32::from_be)
}

/// Without `ip`: IPv6 from /proc/net/if_inet6; IPv4 from /proc/net/fib_trie (local addresses) matched to the
/// directly connected routes of /proc/net/route (Python asks the kernel with SIOCGIFADDR, an ioctl this crate
/// cannot make without `unsafe`: PORTING.md DV20).
pub fn fallback_addresses(paths: &Paths) -> Vec<Value> {
    let mut out = Vec::new();
    let inet6 = read_text(&paths.proc.join("net").join("if_inet6"), 1 << 20).unwrap_or_default();
    for line in py::splitlines(&inet6) {
        let p = py::split_ws(line);
        if p.len() < 6 || skipped(p[5]) || p[0].len() != 32 {
            continue;
        }
        let Ok(bytes) = hex::decode(p[0]) else {
            continue;
        };
        let Ok(arr) = <[u8; 16]>::try_from(bytes.as_slice()) else {
            continue;
        };
        let Ok(prefix) = i64::from_str_radix(p[2], 16) else {
            continue;
        };
        let addr = std::net::Ipv6Addr::from(arr).to_string();
        if usable(&addr) {
            out.push(json!({"iface": py::take(p[5], 32), "addr": addr, "prefix": prefix}));
        }
    }
    // IPv4: connected routes per interface, local addresses from the FIB
    let route = read_text(&paths.proc.join("net").join("route"), 1 << 20).unwrap_or_default();
    let mut nets: Vec<(String, u32, u32)> = Vec::new();
    for line in py::splitlines(&route).into_iter().skip(1) {
        let f = py::split_ws(line);
        if f.len() < 8 {
            continue;
        }
        if let (Some(dest), Some(gw), Some(mask)) =
            (hex_le_ipv4(f[1]), hex_le_ipv4(f[2]), hex_le_ipv4(f[7]))
            && gw == 0
            && mask != 0
        {
            nets.push((f[0].to_string(), dest, mask));
        }
    }
    let fib = read_text(&paths.proc.join("net").join("fib_trie"), 4 << 20).unwrap_or_default();
    let mut locals: Vec<u32> = Vec::new();
    let mut last: Option<u32> = None;
    for line in py::splitlines(&fib) {
        let t = py::strip(line);
        if let Some(a) = t.strip_prefix("|-- ").or_else(|| t.strip_prefix("+-- ")) {
            last = py::split_ws(a)
                .first()
                .and_then(|s| s.parse::<std::net::Ipv4Addr>().ok())
                .map(u32::from);
        } else if t.contains("/32 host LOCAL")
            && let Some(a) = last
            && !locals.contains(&a)
        {
            locals.push(a);
        }
    }
    locals.sort_unstable();
    let mut names: Vec<String> = std::fs::read_dir(paths.sys.join("class").join("net"))
        .map(|d| {
            d.filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    for name in names {
        if skipped(&name) || name.len() > 15 {
            continue;
        }
        let hit = locals.iter().find_map(|a| {
            nets.iter()
                .filter(|(i, net, mask)| *i == name && a & mask == *net)
                .max_by_key(|(_, _, mask)| mask.count_ones())
                .map(|(_, _, mask)| (*a, mask.count_ones()))
        });
        if let Some((a, prefix)) = hit {
            let addr = std::net::Ipv4Addr::from(a).to_string();
            if usable(&addr) {
                out.push(json!({"iface": name, "addr": addr, "prefix": prefix}));
            }
        }
    }
    out
}
