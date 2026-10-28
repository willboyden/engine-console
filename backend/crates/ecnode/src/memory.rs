//! Python: node/src/ecnode/memory.py.
//!
//! Host memory and CPU from /proc, and cgroup v2 memory/CPU of a container through its init pid. Why cgroups: an
//! engine can hold tens of GiB of shared memory and page cache that no single process's RSS shows; `docker stats`
//! reads the same cgroup counters. RSS of the process tree is only the fallback (source "rss"). Roots are
//! injectable (Paths) so tests use fake /proc and /sys/fs/cgroup trees.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use ec_proto::Obj;
use serde_json::{Value, json};

use crate::hostinfo::{Paths, read_text};
use crate::util::py;

static CGROUP_PATH: LazyLock<regex::Regex> =
    LazyLock::new(|| crate::policy::full(r"/[A-Za-z0-9_.@:\-/]{0,400}"));
/// A docker container's cgroup, as /proc/<pid>/cgroup shows it (rootless/rootful systemd driver, cgroupfs driver).
static CONTAINER_ID: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
    fancy_regex::Regex::new(r"/docker[-/]([0-9a-f]{64})(?:\.scope)?(?=/|$)")
        .unwrap_or_else(|_| std::process::abort())
});
pub const MAX_PROCS: usize = 20000;
pub const MEM_KEYS: &[(&str, &str)] = &[
    ("MemTotal", "total"),
    ("MemAvailable", "available"),
    ("MemFree", "free"),
    ("Cached", "cached"),
    ("Shmem", "shmem"),
    ("SwapTotal", "swap_total"),
    ("SwapFree", "swap_free"),
];

pub struct Mem {
    pub paths: Paths,
    cpu_prev: Mutex<Option<(i64, i64)>>,
}

fn int_digits(s: &str) -> Option<i64> {
    let t = py::strip(s);
    if py::isdigit(t) { t.parse().ok() } else { None }
}

impl Mem {
    pub fn new(paths: Paths) -> Self {
        Self {
            paths,
            cpu_prev: Mutex::new(None),
        }
    }

    fn proc_file(&self, pid: i64, name: &str) -> PathBuf {
        self.paths.proc.join(pid.to_string()).join(name)
    }

    // ── host ─────────────────────────────────────────────────────────────────────────────────────────────
    pub fn meminfo(&self) -> Obj {
        let mut out = Obj::new();
        for (_, v) in MEM_KEYS {
            out.insert((*v).to_string(), Value::Null);
        }
        let text = read_text(&self.paths.proc.join("meminfo"), 65536).unwrap_or_default();
        for line in py::splitlines(&text) {
            let (key, rest) = line.split_once(':').unwrap_or((line, ""));
            if let Some((_, name)) = MEM_KEYS.iter().find(|(k, _)| *k == key) {
                let parts = py::split_ws(rest);
                if let Some(p) = parts.first()
                    && py::isdigit(p)
                    && let Ok(n) = p.parse::<i64>()
                {
                    out.insert((*name).to_string(), json!(n * 1024));
                }
            }
        }
        out
    }

    pub fn loadavg(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        let text = read_text(&self.paths.proc.join("loadavg"), 1024).unwrap_or_default();
        let vals: Vec<Option<f64>> = py::split_ws(&text)
            .into_iter()
            .take(3)
            .map(py::parse_float)
            .collect();
        let get = |i: usize| vals.get(i).copied().flatten();
        (get(0), get(1), get(2))
    }

    /// Percent busy across all CPUs since the previous call (None on the first call).
    pub fn cpu_util(&self) -> Option<f64> {
        let text = read_text(&self.paths.proc.join("stat"), 8192).unwrap_or_default();
        let line = text.split('\n').next().unwrap_or("");
        let fields = py::split_ws(line);
        if fields.len() < 5 || fields[0] != "cpu" {
            return None;
        }
        let mut nums = Vec::new();
        for x in fields.iter().skip(1).take(8) {
            nums.push(py::parse_int(x)?);
        }
        let idle = nums[3] + nums.get(4).copied().unwrap_or(0);
        let total: i64 = nums.iter().sum();
        let mut prev_slot = crate::util::lock(&self.cpu_prev);
        let prev = prev_slot.replace((total, idle));
        let (pt, pi) = prev?;
        if total <= pt {
            return None;
        }
        let busy = 1.0 - (idle - pi) as f64 / (total - pt) as f64;
        Some(py::round(busy.clamp(0.0, 1.0) * 100.0, 1))
    }

    // ── processes ────────────────────────────────────────────────────────────────────────────────────────
    /// The process name (comm). Never the command line: GPU boxes often carry API keys in argv.
    pub fn comm(&self, pid: i64) -> Option<String> {
        let text = read_text(&self.proc_file(pid, "comm"), 256)?;
        if text.is_empty() {
            return None;
        }
        Some(py::take(py::strip(&text), 64).to_string())
    }

    /// The 64-hex docker container id a process belongs to, from /proc/<pid>/cgroup (the outermost one when
    /// containers are nested), or None for a process outside docker.
    pub fn container_id_of(&self, pid: i64) -> Option<String> {
        let text = read_text(&self.proc_file(pid, "cgroup"), 65536).unwrap_or_default();
        for line in py::splitlines(&text) {
            let path = py::strip(line.splitn(3, ':').last().unwrap_or(""));
            if let Ok(Some(c)) = CONTAINER_ID.captures(path)
                && let Some(m) = c.get(1)
            {
                return Some(m.as_str().to_string());
            }
        }
        None
    }

    // ── containers ───────────────────────────────────────────────────────────────────────────────────────
    pub fn cgroup_dir(&self, pid: i64) -> Option<PathBuf> {
        let text = read_text(&self.proc_file(pid, "cgroup"), 65536)?;
        let line = py::splitlines(&text)
            .into_iter()
            .find(|l| l.starts_with("0::"))?;
        let path = py::strip(&line[3..]);
        // untrusted text: strict charset, no traversal, and the resolved dir must stay under the cgroup root
        if !CGROUP_PATH.is_match(path) || path.split('/').any(|s| s == "..") {
            return None;
        }
        let base = py::realpath(&self.paths.cgroup());
        let target = py::realpath(&base.join(path.trim_start_matches('/')));
        if target != base && !target.starts_with(&base) {
            return None;
        }
        target.is_dir().then_some(target)
    }

    /// CG_MEM of the container whose init process is `pid` (cgroup v2, else summed RSS), or None.
    pub fn container_mem(&self, pid: i64) -> Option<Obj> {
        if pid <= 0 {
            return None;
        }
        if let Some(d) = self.cgroup_dir(pid) {
            let current = read_text(&d.join("memory.current"), 64);
            let stat_text = read_text(&d.join("memory.stat"), 65536);
            if let (Some(cur), Some(st)) = (current.as_deref().and_then(int_digits), stat_text) {
                let mut stat: HashMap<String, i64> = HashMap::new();
                for line in py::splitlines(&st) {
                    let (k, v) = line.split_once(' ').unwrap_or((line, ""));
                    if let Some(n) = int_digits(v) {
                        stat.insert(k.to_string(), n);
                    }
                }
                let g = |k: &str| stat.get(k).copied().unwrap_or(0);
                let kernel = stat.get("kernel").copied().unwrap_or_else(|| {
                    g("kernel_stack") + g("slab") + g("pagetables") + g("percpu")
                });
                return Some(obj(json!({
                    "current": cur, "anon": g("anon"), "file": g("file"), "shmem": g("shmem"),
                    "kernel": kernel, "source": "cgroup",
                })));
            }
        }
        let rss = self.rss_tree(pid)?;
        Some(obj(json!({
            "current": rss, "anon": rss, "file": 0, "shmem": 0, "kernel": 0, "source": "rss",
        })))
    }

    pub fn container_cpu_usec(&self, pid: i64) -> Option<i64> {
        let d = if pid > 0 {
            self.cgroup_dir(pid)?
        } else {
            return None;
        };
        let text = read_text(&d.join("cpu.stat"), 8192).unwrap_or_default();
        for line in py::splitlines(&text) {
            let (k, v) = line.split_once(' ').unwrap_or((line, ""));
            if k == "usage_usec"
                && let Some(n) = int_digits(v)
            {
                return Some(n);
            }
        }
        None
    }

    /// Sum of VmRSS over the process tree under pid (underestimates: misses shared memory and cache).
    fn rss_tree(&self, pid: i64) -> Option<i64> {
        let mut children: HashMap<i64, Vec<i64>> = HashMap::new();
        let entries = std::fs::read_dir(&self.paths.proc).ok()?;
        for (i, entry) in entries.enumerate() {
            if i > MAX_PROCS {
                break;
            }
            let Ok(entry) = entry else { continue };
            let name = entry.file_name().to_string_lossy().into_owned();
            if !py::isdigit(&name) {
                continue;
            }
            let Some(stat) = read_text(&entry.path().join("stat"), 4096) else {
                continue;
            };
            let Some(close) = stat.rfind(')') else {
                continue;
            };
            let Some(ppid) = py::split_ws(&stat[close + 1..])
                .get(1)
                .and_then(|p| py::parse_int(p))
            else {
                continue;
            };
            let Ok(me) = name.parse::<i64>() else {
                continue;
            };
            children.entry(ppid).or_default().push(me);
        }
        if !self.paths.proc.join(pid.to_string()).is_dir() {
            return None;
        }
        let mut seen: HashSet<i64> = HashSet::from([pid]);
        let mut stack = vec![pid];
        let mut total_kb: i64 = 0;
        while let Some(cur) = stack.pop() {
            let text = read_text(&self.proc_file(cur, "status"), 65536).unwrap_or_default();
            for line in py::splitlines(&text) {
                if line.starts_with("VmRSS:") {
                    let parts = py::split_ws(line);
                    if parts.len() >= 2 && py::isdigit(parts[1]) {
                        total_kb += parts[1].parse::<i64>().unwrap_or(0);
                    }
                    break;
                }
            }
            for ch in children.get(&cur).map(Vec::as_slice).unwrap_or_default() {
                if seen.insert(*ch) {
                    stack.push(*ch);
                }
            }
        }
        Some(total_kb * 1024)
    }
}

fn obj(v: Value) -> Obj {
    match v {
        Value::Object(o) => o,
        _ => Obj::new(),
    }
}

/// `Path.resolve()` of the cgroup root (for tests that need the same base).
pub fn cgroup_base(paths: &Paths) -> PathBuf {
    py::realpath(Path::new(&paths.cgroup()))
}
