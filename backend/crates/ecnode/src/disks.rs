//! Python: node/src/ecnode/disks.py.
//!
//! Filesystems relevant to this node: `/`, the filesystem of every model root, and every network mount. Mounts come
//! from /proc/self/mountinfo, sizes from statvfs, the drive model from /sys/block/<disk>/device/model. NFS/CIFS
//! sources are split into remote_host/remote_path so the hub can tell when two machines see the same storage.
//! statvfs and realpath on a dead network mount can block forever, so they run through util::Guarded.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use ec_proto::Obj;
use serde_json::{Value, json};

use crate::hostinfo::{Paths, read_text};
use crate::policy::Root;
use crate::util::{Guarded, clean_text, py};

pub const PSEUDO_FS: &[&str] = &[
    "proc",
    "sysfs",
    "devtmpfs",
    "devpts",
    "tmpfs",
    "cgroup",
    "cgroup2",
    "securityfs",
    "debugfs",
    "tracefs",
    "configfs",
    "fusectl",
    "mqueue",
    "hugetlbfs",
    "pstore",
    "bpf",
    "autofs",
    "binfmt_misc",
    "efivarfs",
    "nsfs",
    "overlay",
    "squashfs",
    "ramfs",
    "rpc_pipefs",
    "nfsd",
    "selinuxfs",
    "fuse.portal",
    "fuse.gvfsd-fuse",
    "fuse.snapfuse",
    "fuse.lxcfs",
    "fuse.xdg-document-portal",
    "devfs",
    "shiftfs",
    "zonefs",
    "iso9660",
];
pub const NETWORK_FS: &[&str] = &[
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "smbfs",
    "ceph",
    "glusterfs",
    "fuse.glusterfs",
    "fuse.sshfs",
    "lustre",
    "beegfs",
    "gpfs",
    "9p",
    "afs",
    "fuse.juicefs",
    "fuse.s3fs",
    "fuse.rclone",
    "fuse.cephfs",
    "virtiofs",
    "davfs",
    "fuse.davfs2",
];
pub const STAT_TIMEOUT_S: f64 = 5.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountEntry {
    pub mount: String,
    pub fstype: String,
    pub source: String,
    pub options: Vec<String>,
    /// "major:minor"
    pub dev: String,
}

/// The statvfs fields the report uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfsStat {
    pub frsize: u64,
    pub blocks: u64,
    pub bfree: u64,
    pub bavail: u64,
}

pub type StatvfsFn = Arc<dyn Fn(&Path) -> std::io::Result<VfsStat> + Send + Sync>;

/// `os.statvfs`
pub fn os_statvfs() -> StatvfsFn {
    Arc::new(|p: &Path| {
        let st = nix::sys::statvfs::statvfs(p).map_err(std::io::Error::from)?;
        Ok(VfsStat {
            frsize: st.fragment_size(),
            blocks: st.blocks(),
            bfree: st.blocks_free(),
            bavail: st.blocks_available(),
        })
    })
}

/// mountinfo escapes space, tab, newline and backslash as \ooo octal (each code becomes one character).
fn unescape(s: &str) -> String {
    if !s.contains('\\') {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\'
            && i + 3 < chars.len()
            && chars[i + 1..i + 4].iter().all(|c| ('0'..='7').contains(c))
        {
            let code: String = chars[i + 1..i + 4].iter().collect();
            if let Some(c) = u32::from_str_radix(&code, 8).ok().and_then(char::from_u32) {
                out.push(c);
                i += 4;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

pub fn parse_mountinfo(text: &str) -> Vec<MountEntry> {
    let mut out = Vec::new();
    for line in py::splitlines(text) {
        let Some((left, right)) = line.split_once(" - ") else {
            continue;
        };
        let (lf, rf) = (py::split_ws(left), py::split_ws(right));
        if lf.len() < 6 || rf.len() < 2 {
            continue;
        }
        out.push(MountEntry {
            mount: unescape(lf[4]),
            fstype: rf[0].to_string(),
            source: unescape(rf[1]),
            options: lf[5].split(',').map(str::to_string).collect(),
            dev: lf[2].to_string(),
        });
    }
    out
}

/// (remote_host, remote_path) of a network filesystem source like host:/export or //host/share.
pub fn split_remote(fstype: &str, source: &str) -> (Option<String>, Option<String>) {
    if fstype.starts_with("cifs") || fstype.starts_with("smb") || source.starts_with("//") {
        let s = source.trim_start_matches('/');
        let (host, rest) = match s.split_once('/') {
            Some((h, r)) => (h, r),
            None => (s, ""),
        };
        return (
            Some(host.to_string()).filter(|h| !h.is_empty()),
            Some(format!("/{rest}")),
        );
    }
    if source.starts_with('[')
        && let Some(end) = source.find(']').filter(|e| *e > 0)
    {
        let path = source[end + 1..].trim_start_matches(':');
        return (
            Some(source[1..end].to_string()),
            Some(if path.is_empty() {
                "/".to_string()
            } else {
                path.to_string()
            }),
        );
    }
    let Some((host, path)) = source.split_once(':') else {
        return (None, None);
    };
    let host = host.rsplit_once('@').map_or(host, |(_, h)| h); // sshfs: user@host:path
    (
        Some(host.to_string()).filter(|h| !h.is_empty()),
        Some(if path.is_empty() {
            "/".to_string()
        } else {
            path.to_string()
        }),
    )
}

/// The mount a path lives on: the longest mount point that is a prefix (the later one when stacked).
pub fn mount_of<'a>(path: &str, mounts: &'a [MountEntry]) -> Option<&'a MountEntry> {
    let mut best: Option<&MountEntry> = None;
    for m in mounts {
        let mp = m.mount.as_str();
        let covers =
            path == mp || mp == "/" || path.starts_with(&format!("{}/", mp.trim_end_matches('/')));
        if covers && best.is_none_or(|b| py::len(mp) >= py::len(&b.mount)) {
            best = Some(m);
        }
    }
    best
}

pub struct Disks {
    pub paths: Paths,
    statvfs: StatvfsFn,
    guard: Guarded,
}

impl Disks {
    pub fn new(paths: Paths) -> Self {
        Self::with_statvfs(paths, os_statvfs())
    }

    pub fn with_statvfs(paths: Paths, statvfs: StatvfsFn) -> Self {
        Self {
            paths,
            statvfs,
            guard: Guarded::new(),
        }
    }

    pub fn mounts(&self) -> Vec<MountEntry> {
        parse_mountinfo(
            &read_text(&self.paths.proc.join("self").join("mountinfo"), 4 << 20)
                .unwrap_or_default(),
        )
    }

    /// realpath of a root (its nearest existing ancestor if it is missing), bounded in time.
    pub fn resolve(&self, path: &str) -> String {
        let p = PathBuf::from(path);
        let real = self.guard.call(
            &format!("realpath:{path}"),
            move || py::realpath(&p),
            Duration::from_secs_f64(STAT_TIMEOUT_S),
        );
        match real {
            Some(r) if !r.as_os_str().is_empty() => py::path_str(&r),
            _ => py::normpath_str(path),
        }
    }

    pub fn report(&self, roots: &[Root]) -> Obj {
        let mounts = self.mounts();
        let mut by_mount: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut chosen: BTreeMap<String, MountEntry> = BTreeMap::new();
        for r in roots {
            if let Some(m) = mount_of(&self.resolve(&r.path), &mounts) {
                chosen.insert(m.mount.clone(), m.clone());
                by_mount
                    .entry(m.mount.clone())
                    .or_default()
                    .push(r.id.clone());
            }
        }
        for m in &mounts {
            if m.mount == "/" || NETWORK_FS.contains(&m.fstype.as_str()) {
                chosen.entry(m.mount.clone()).or_insert_with(|| m.clone());
            }
        }
        // a stacked mount point: keep the entry that is on top (the last one in mountinfo)
        let mut top: BTreeMap<&str, &MountEntry> = BTreeMap::new();
        for m in &mounts {
            top.insert(m.mount.as_str(), m);
        }
        // sorted(chosen.items()): by mount point, as Python compares strings (code points; BTreeMap sorts bytes,
        // which is the same order for UTF-8)
        let out: Vec<Value> = chosen
            .iter()
            .take(64)
            .map(|(mp, m)| {
                let m = top.get(mp.as_str()).copied().unwrap_or(m);
                Value::Object(
                    self.describe(m, by_mount.get(mp).map(Vec::as_slice).unwrap_or_default()),
                )
            })
            .collect();
        let mut o = Obj::new();
        o.insert("t".into(), json!(crate::agent::now3()));
        o.insert("mounts".into(), Value::Array(out));
        o
    }

    fn describe(&self, m: &MountEntry, roots: &[String]) -> Obj {
        let remote = NETWORK_FS.contains(&m.fstype.as_str());
        let (rhost, rpath) = if remote {
            split_remote(&m.fstype, &m.source)
        } else {
            (None, None)
        };
        let f = Arc::clone(&self.statvfs);
        let mp = PathBuf::from(&m.mount);
        let st = self
            .guard
            .call(
                &format!("statvfs:{}", m.mount),
                move || f(&mp).ok(),
                Duration::from_secs_f64(STAT_TIMEOUT_S),
            )
            .flatten();
        let (size, used, free) = match st {
            Some(s) => (
                Some(s.blocks.saturating_mul(s.frsize)),
                Some(s.blocks.saturating_sub(s.bfree).saturating_mul(s.frsize)),
                Some(s.bavail.saturating_mul(s.frsize)),
            ),
            None => (None, None, None),
        };
        let (device, model) = self.device(m);
        let v = json!({
            "mount": clean_text(&m.mount, 512), "fstype": clean_text(&m.fstype, 32),
            "source": clean_text(&m.source, 512), "device": device, "model": model, "size": size, "used": used,
            "free": free, "readonly": m.options.iter().any(|o| o == "ro"), "remote": remote,
            "remote_host": rhost.filter(|h| !h.is_empty()).map(|h| clean_text(&h, 253)),
            "remote_path": rpath.filter(|p| !p.is_empty()).map(|p| clean_text(&p, 512)),
            "roots": roots,
        });
        match v {
            Value::Object(o) => o,
            _ => Obj::new(),
        }
    }

    fn device(&self, m: &MountEntry) -> (Option<String>, Option<String>) {
        let Some(dev) = m.source.strip_prefix("/dev/") else {
            return (None, None);
        };
        (Some(clean_text(dev, 64)), self.model(&m.dev, 0))
    }

    /// Drive model of a block device number, through partitions and (up to 3 levels of) dm/md slaves.
    fn model(&self, dev: &str, depth: usize) -> Option<String> {
        if depth > 3 || !dev.contains(':') {
            return None;
        }
        let node = py::realpath(&self.paths.sys.join("dev").join("block").join(dev));
        let name_of = |p: &Path| py::basename(p).to_string_lossy().into_owned();
        let disk = if node.join("partition").exists() {
            name_of(&py::dirname(&node))
        } else {
            name_of(&node)
        };
        let block = self.paths.sys.join("block").join(&disk);
        if let Some(text) = read_text(&block.join("device").join("model"), 256) {
            let t = py::strip(&text);
            if !t.is_empty() {
                return Some(clean_text(t, 128));
            }
        }
        let mut slaves: Vec<String> = std::fs::read_dir(block.join("slaves"))
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect();
        slaves.sort();
        for s in slaves.iter().take(4) {
            let num = read_text(
                &self
                    .paths
                    .sys
                    .join("class")
                    .join("block")
                    .join(s)
                    .join("dev"),
                32,
            )
            .unwrap_or_default();
            let num = py::strip(&num);
            if !num.is_empty()
                && let Some(found) = self.model(num, depth + 1)
            {
                return Some(found);
            }
        }
        None
    }
}
