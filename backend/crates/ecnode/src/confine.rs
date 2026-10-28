//! Python: node/src/ecnode/confine.py.
//!
//! Path rules for every (root, rel) a hub names (docs/FLEET-PROTOCOL.md §4 "Path rules").
//!
//! `rel` is relative: no leading "/", no "\\", no NUL or other control character, no empty, "." or ".." segment.
//! The parent directory is resolved with realpath and must lie inside the root's own realpath; the directory is
//! then opened and its fd re-checked through /proc/self/fd, so a directory swapped for a symlink in between is
//! caught. Final components are opened relative to that fd with O_NOFOLLOW.
//!
//! Paths are handled as bytes (`Path`), never as lossy text, so `inside` compares exactly what the kernel sees.

use std::ffi::OsStr;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use ec_proto::{BAD_ARGS, IO_ERROR, NO_SPACE, NOT_FOUND, POLICY_DENIED};
use nix::errno::Errno;
use nix::fcntl::{AtFlags, OFlag};
use nix::sys::stat::{FileStat, Mode};
use serde_json::Value;

use crate::errors::{HandlerError, OpError};
use crate::policy::{Policy, Root};
use crate::util::{is_secret_name, is_write_protected_name, py};

pub const PROC_SELF_FD: &str = "/proc/self/fd";

/// `os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC`
pub fn dir_flags() -> OFlag {
    OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC
}

static BAD_CHARS: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"[\x00-\x1f\x7f\\]").unwrap_or_else(|_| std::process::abort())
});

/// Either a refusal (OpError) or an OSError that Python lets propagate to the server (which answers `internal`
/// naming the exception class).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fail {
    Op(OpError),
    /// errno of an OSError nobody caught
    Os(i32),
}

impl From<OpError> for Fail {
    fn from(e: OpError) -> Self {
        Fail::Op(e)
    }
}

impl From<Errno> for Fail {
    fn from(e: Errno) -> Self {
        Fail::Os(e as i32)
    }
}

impl From<Fail> for HandlerError {
    fn from(f: Fail) -> Self {
        match f {
            Fail::Op(e) => HandlerError::Op(e),
            Fail::Os(n) => HandlerError::internal(py::os_error_name(n)),
        }
    }
}

fn op(code: &str, msg: impl Into<String>) -> OpError {
    OpError::new(code, msg)
}

/// `os.strerror(errno)` / `e.strerror`: the C library's text, as Python shows it (std asks libc; nix's own table
/// words some errors differently, e.g. ELOOP).
pub fn strerror(errno: i32) -> String {
    let text = std::io::Error::from_raw_os_error(errno).to_string();
    match text.strip_suffix(&format!(" (os error {errno})")) {
        Some(t) => t.to_string(),
        None => text,
    }
}

fn desc(e: Errno) -> String {
    strerror(e as i32)
}

/// The segments of a valid relative path, else OpError(bad_args).
pub fn split_rel(rel: Option<&str>, what: &str) -> Result<Vec<String>, OpError> {
    let rel = match rel {
        Some(r) if !r.is_empty() => r,
        _ => {
            return Err(op(
                BAD_ARGS,
                format!("the {what} is empty; name a file or directory inside the model root"),
            ));
        }
    };
    if rel.starts_with('/') {
        return Err(op(
            BAD_ARGS,
            format!(
                "the {what} {} must be relative to the model root, not absolute",
                py::repr_take(rel, 200)
            ),
        ));
    }
    if BAD_CHARS.is_match(rel) {
        return Err(op(
            BAD_ARGS,
            format!("the {what} contains a backslash or a control character"),
        ));
    }
    let parts: Vec<String> = rel.split('/').map(str::to_string).collect();
    for p in &parts {
        if p.is_empty() || p == "." || p == ".." {
            return Err(op(
                BAD_ARGS,
                format!(
                    "the {what} {} has an empty, '.' or '..' segment",
                    py::repr_take(rel, 200)
                ),
            ));
        }
        if p.len() > 255 {
            return Err(op(
                BAD_ARGS,
                format!("the {what} has a segment longer than 255 bytes"),
            ));
        }
    }
    Ok(parts)
}

/// `child == root or child.startswith(root.rstrip("/") + "/")` on the bytes.
pub fn inside(child: &Path, root: &Path) -> bool {
    let c = child.as_os_str().as_bytes();
    let r = root.as_os_str().as_bytes();
    if c == r {
        return true;
    }
    let mut end = r.len();
    while end > 0 && r[end - 1] == b'/' {
        end -= 1;
    }
    c.len() > end && c.starts_with(&r[..end]) && c[end] == b'/'
}

/// True when any segment of `real` below the root looks like a key or credential store, so a symlink inside the
/// root cannot lead a read to a secret directory that also sits inside the root.
pub fn secret_inside(real: &Path, root_real: &Path) -> bool {
    any_segment_below(real, root_real, is_secret_name)
}

/// True when any segment of `real` below the root is a name writes refuse (`is_write_protected_name`).
pub fn write_protected_inside(real: &Path, root_real: &Path) -> bool {
    any_segment_below(real, root_real, is_write_protected_name)
}

fn any_segment_below(real: &Path, root_real: &Path, hit: fn(&str) -> bool) -> bool {
    let rel = if inside(real, root_real) {
        py::relpath(real, root_real)
    } else {
        real.to_path_buf()
    };
    rel.as_os_str()
        .as_bytes()
        .split(|c| *c == b'/')
        .filter(|s| !s.is_empty() && *s != b".")
        .any(|s| hit(&String::from_utf8_lossy(s)))
}

/// What a write refuses when a segment of its path has a protected name.
pub const WRITE_PROTECTED: &str = "refusing to write a file or directory whose name looks like a key, a credential, \
                                   a login script or git's settings";

/// split_rel for a path the node creates or replaces (fs.write, fs.put_small, fs.symlink): every segment must also
/// pass `is_write_protected_name` (security audit F2, beyond Python: reads already refuse secret-looking names).
pub fn split_rel_for_write(rel: Option<&str>, what: &str) -> Result<Vec<String>, OpError> {
    let parts = split_rel(rel, what)?;
    if parts.iter().any(|p| is_write_protected_name(p)) {
        return Err(op(POLICY_DENIED, WRITE_PROTECTED));
    }
    Ok(parts)
}

/// A directory a write goes into must not be, or lie below, a write-protected name inside the root: a symlink
/// inside the root can lead there although no segment of the path names it (security audit F2).
fn check_write_dir(fd: &OwnedFd, h: &RootHandle) -> Result<(), OpError> {
    match fd_path(fd.as_fd()) {
        Some(w) if !write_protected_inside(&w, &h.real) => Ok(()),
        Some(_) => Err(op(POLICY_DENIED, WRITE_PROTECTED)),
        None => Err(op(POLICY_DENIED, "that path leads outside the model root")),
    }
}

/// Where an fd really points (`/proc/self/fd/N`), None when that cannot be read.
pub fn fd_path(fd: BorrowedFd<'_>) -> Option<PathBuf> {
    use std::os::fd::AsRawFd;
    std::fs::read_link(format!("{PROC_SELF_FD}/{}", fd.as_raw_fd())).ok()
}

/// A root of the policy and its realpath.
#[derive(Debug, Clone, PartialEq)]
pub struct RootHandle {
    pub root: Root,
    /// realpath of the root directory
    pub real: PathBuf,
}

impl RootHandle {
    /// `os.path.join(real, *parts)`
    pub fn path<S: AsRef<OsStr>>(&self, parts: &[S]) -> PathBuf {
        py::join(&self.real, parts)
    }
}

/// The root named by `root_id` (a cleaned argument: a string, or null when it was not one).
pub fn open_root(policy: &Policy, root_id: &Value, writable: bool) -> Result<RootHandle, OpError> {
    let Some(root) = policy.root_of(root_id) else {
        return Err(op(
            NOT_FOUND,
            format!(
                "there is no model root {} in this node's policy ({})",
                py::repr_take(&py::str_of(root_id), 32),
                policy.where_()
            ),
        ));
    };
    if writable && !root.writable {
        return Err(op(
            POLICY_DENIED,
            format!(
                "model root {} is read-only in this node's policy; an administrator can set \"writable\": true \
                 in {}",
                py::repr(&root.id),
                policy.where_()
            ),
        ));
    }
    let real = py::realpath(Path::new(&root.path));
    if !real.is_dir() {
        return Err(op(
            NOT_FOUND,
            format!(
                "model root {} ({}) does not exist on this machine (not mounted?)",
                py::repr(&root.id),
                root.path
            ),
        ));
    }
    Ok(RootHandle {
        root: root.clone(),
        real,
    })
}

fn check_fd(fd: OwnedFd, h: &RootHandle) -> Result<OwnedFd, OpError> {
    check_where(fd_path(fd.as_fd()).as_deref(), h)?;
    Ok(fd)
}

/// An fd's real location (`fd_path`) must be inside the root. One that cannot be read (None) is refused too
/// (security audit F4, beyond Python, which let it pass): fs.delete already refused it.
fn check_where(where_: Option<&Path>, h: &RootHandle) -> Result<(), OpError> {
    match where_ {
        Some(w) if inside(w, &h.real) => Ok(()),
        _ => Err(op(POLICY_DENIED, "that path leads outside the model root")),
    }
}

fn joined(parts: &[String]) -> String {
    py::repr_take(&parts.join("/"), 300)
}

/// A directory fd for root/parts. Each existing component may be a symlink only if it resolves inside the root.
/// With create=true missing directories are made (0755).
pub fn open_dir(h: &RootHandle, parts: &[String], create: bool) -> Result<OwnedFd, Fail> {
    if !create {
        let real = py::realpath(&h.path(parts));
        if !inside(&real, &h.real) {
            return Err(op(
                POLICY_DENIED,
                "that path resolves to somewhere outside the model root",
            )
            .into());
        }
        let fd = match nix::fcntl::open(&real, dir_flags(), Mode::empty()) {
            Ok(fd) => fd,
            Err(Errno::ENOENT) => {
                return Err(op(
                    NOT_FOUND,
                    format!(
                        "directory {} does not exist in the model root",
                        joined(parts)
                    ),
                )
                .into());
            }
            Err(Errno::ENOTDIR) => {
                return Err(op(BAD_ARGS, format!("{} is not a directory", joined(parts))).into());
            }
            Err(e) => {
                return Err(op(
                    IO_ERROR,
                    format!("cannot open {}: {}", joined(parts), desc(e)),
                )
                .into());
            }
        };
        return Ok(check_fd(fd, h)?);
    }
    // the root itself: an OSError here is not caught in Python either
    let mut fd = nix::fcntl::open(&h.real, dir_flags(), Mode::empty())?;
    let step = |fd: &OwnedFd, p: &str| -> Result<OwnedFd, Fail> {
        match nix::fcntl::openat(fd.as_fd(), p, dir_flags(), Mode::empty()) {
            Ok(n) => Ok(n),
            Err(Errno::ENOENT) => {
                match nix::sys::stat::mkdirat(fd.as_fd(), p, Mode::from_bits_truncate(0o755)) {
                    Ok(()) | Err(Errno::EEXIST) => {}
                    Err(e) => return Err(Fail::Os(e as i32)),
                }
                nix::fcntl::openat(fd.as_fd(), p, dir_flags(), Mode::empty())
                    .map_err(|e| Fail::Os(e as i32))
            }
            Err(Errno::ELOOP | Errno::ENOTDIR) => {
                // a symlinked directory: allowed only when it resolves inside the root
                let base = fd_path(fd.as_fd()).unwrap_or_else(|| h.real.clone());
                let real = py::realpath(&py::join(&base, &[p]));
                if !inside(&real, &h.real) || !real.is_dir() {
                    return Err(op(
                        POLICY_DENIED,
                        format!(
                            "{} is a link to somewhere outside the model root, or not a directory",
                            py::repr(p)
                        ),
                    )
                    .into());
                }
                nix::fcntl::open(&real, dir_flags(), Mode::empty()).map_err(|e| Fail::Os(e as i32))
            }
            Err(e) => Err(Fail::Os(e as i32)),
        }
    };
    // create=true is a write (fs.write, fs.put_small, fs.symlink): every directory on the way is checked before
    // anything is created in it
    for p in parts {
        match step(&fd, p) {
            Ok(n) => {
                fd = check_fd(n, h)?;
                check_write_dir(&fd, h)?;
            }
            Err(Fail::Op(e)) => return Err(Fail::Op(e)),
            Err(Fail::Os(n)) => {
                if n == Errno::ENOSPC as i32 {
                    return Err(op(NO_SPACE, "the disk is full").into());
                }
                return Err(op(
                    IO_ERROR,
                    format!("cannot create directories for that path: {}", strerror(n)),
                )
                .into());
            }
        }
    }
    Ok(fd)
}

/// `os.stat(name, dir_fd=dfd, follow_symlinks=False)`; None when it does not exist.
pub fn lstat_at(dfd: BorrowedFd<'_>, name: &str) -> Result<Option<FileStat>, OpError> {
    lstat_at_os(dfd, OsStr::new(name))
}

/// `lstat_at` for a name that may not be UTF-8 (a directory listing during fs.delete).
pub fn lstat_at_os(dfd: BorrowedFd<'_>, name: &OsStr) -> Result<Option<FileStat>, OpError> {
    match nix::sys::stat::fstatat(dfd, name, AtFlags::AT_SYMLINK_NOFOLLOW) {
        Ok(st) => Ok(Some(st)),
        Err(Errno::ENOENT) => Ok(None),
        Err(e) => Err(op(
            IO_ERROR,
            format!(
                "cannot look at {}: {}",
                py::repr_take(&name.to_string_lossy(), 200),
                desc(e)
            ),
        )),
    }
}

/// Realpath of a symlink at dfd/name when it resolves inside the root, else None.
pub fn resolve_link_inside(dfd: BorrowedFd<'_>, name: &str, h: &RootHandle) -> Option<PathBuf> {
    let base = fd_path(dfd)?;
    let real = py::realpath(&py::join(&base, &[name]));
    inside(&real, &h.real).then_some(real)
}

/// Realpath of root/parts (it must exist and stay inside the root), for mounts and in-use checks.
pub fn resolve_existing(h: &RootHandle, parts: &[String]) -> Result<PathBuf, OpError> {
    let real = py::realpath(&h.path(parts));
    if !inside(&real, &h.real) {
        return Err(op(
            POLICY_DENIED,
            "that path resolves to somewhere outside the model root",
        ));
    }
    if std::fs::metadata(&real).is_err() {
        return Err(op(
            NOT_FOUND,
            format!(
                "{} does not exist in model root {}",
                joined(parts),
                py::repr(&h.root.id)
            ),
        ));
    }
    Ok(real)
}

/// Bytes of a small metadata file (config.json, refs/main, a manifest) when its realpath is inside the root, it
/// is a regular file of at most max_bytes and its name does not look like a secret; else None.
pub fn read_inside(root_real: &Path, path: &Path, max_bytes: usize) -> Option<Vec<u8>> {
    let fd = open_inside(root_real, path)?;
    let st = nix::sys::stat::fstat(fd.as_fd()).ok()?;
    if usize::try_from(st.st_size).unwrap_or(usize::MAX) > max_bytes {
        return None;
    }
    let mut f = std::fs::File::from(fd);
    let mut buf = Vec::new();
    let mut chunk = vec![0u8; 65536];
    while buf.len() <= max_bytes {
        let want = (max_bytes + 1 - buf.len()).min(chunk.len());
        match std::io::Read::read(&mut f, &mut chunk[..want]) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
    (buf.len() <= max_bytes).then_some(buf)
}

/// An fd of a regular file whose realpath is inside root_real (symlinks inside the root are followed, as in an
/// HF snapshot), never a file with a secret-looking name; None otherwise. Never blocks on a FIFO.
pub fn open_inside(root_real: &Path, path: &Path) -> Option<OwnedFd> {
    if path.as_os_str().as_bytes().contains(&0) {
        return None;
    }
    let real = py::realpath(path);
    if !inside(&real, root_real)
        || secret_inside(&real, root_real)
        || is_secret_name(&py::basename(path).to_string_lossy())
    {
        return None;
    }
    // not even opened unless it is a regular file now (opening a device can have side effects); the fstat below
    // re-checks the file actually opened (node hardening)
    let lst = nix::sys::stat::lstat(&real).ok()?;
    if !crate::util::fsread::is_regular(&lst) {
        return None;
    }
    let fd = nix::fcntl::open(
        &real,
        OFlag::O_RDONLY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC | OFlag::O_NONBLOCK,
        Mode::empty(),
    )
    .ok()?;
    let st = nix::sys::stat::fstat(fd.as_fd()).ok()?;
    if st.st_mode & nix::libc::S_IFMT != nix::libc::S_IFREG
        || !fd_inside(fd_path(fd.as_fd()).as_deref(), root_real)
    {
        return None;
    }
    Some(fd)
}

/// open_inside's re-check of the opened file: its real location must be known and inside the root (a failed
/// readlink of /proc/self/fd counts as outside: security audit F4, beyond Python).
fn fd_inside(where_: Option<&Path>, root_real: &Path) -> bool {
    where_.is_some_and(|w| inside(w, root_real))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_compares_whole_segments() {
        assert!(inside(Path::new("/a/b"), Path::new("/a")));
        assert!(inside(Path::new("/a"), Path::new("/a")));
        assert!(inside(Path::new("/a/b"), Path::new("/a/")));
        assert!(!inside(Path::new("/ab"), Path::new("/a")));
        assert!(inside(Path::new("/x"), Path::new("/")));
        assert!(!inside(Path::new("/"), Path::new("/a")));
        // Python's quirk, kept: "/a" is not inside "/a/"
        assert!(!inside(Path::new("/a"), Path::new("/a/")));
    }

    fn handle(root: &Path, writable: bool) -> (Policy, RootHandle) {
        let policy = Policy {
            path: Some("/x/policy.json".into()),
            model_roots: vec![Root {
                writable,
                ..Root::new("t", &root.to_string_lossy())
            }],
            ..Policy::default()
        };
        let h = open_root(&policy, &Value::from("t"), false).unwrap_or_else(|e| panic!("{e}"));
        (policy, h)
    }

    #[test]
    fn test_wsd_check_fd_refuses_an_fd_outside_the_root() {
        // the /proc/self/fd re-check: a directory swapped for a link between realpath and open is caught here
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let root = tmp.path().join("root");
        std::fs::create_dir_all(&root).unwrap_or_else(|e| panic!("{e}"));
        let (_, h) = handle(&root, true);
        let outside = nix::fcntl::open(tmp.path(), dir_flags(), Mode::empty())
            .unwrap_or_else(|e| panic!("{e}"));
        let e = check_fd(outside, &h)
            .err()
            .unwrap_or_else(|| panic!("an outside fd passed"));
        assert_eq!(
            (e.code.as_str(), e.message.as_str()),
            (POLICY_DENIED, "that path leads outside the model root")
        );
        let inner =
            nix::fcntl::open(&root, dir_flags(), Mode::empty()).unwrap_or_else(|e| panic!("{e}"));
        assert!(check_fd(inner, &h).is_ok());
    }

    #[test]
    fn test_audit_f4_an_fd_whose_location_cannot_be_read_is_refused() {
        // security audit F4: readlink of /proc/self/fd/N failing (None) used to pass both re-checks
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let root = tmp.path().join("root");
        std::fs::create_dir_all(&root).unwrap_or_else(|e| panic!("{e}"));
        let (_, h) = handle(&root, true);
        let e = check_where(None, &h)
            .err()
            .unwrap_or_else(|| panic!("an unknown location passed check_fd"));
        assert_eq!(
            (e.code.as_str(), e.message.as_str()),
            (POLICY_DENIED, "that path leads outside the model root")
        );
        assert!(check_where(Some(&h.real.join("x")), &h).is_ok());
        assert!(check_where(Some(tmp.path()), &h).is_err());
        assert!(!fd_inside(None, &h.real));
        assert!(fd_inside(Some(&h.real.join("f")), &h.real));
        assert!(!fd_inside(Some(tmp.path()), &h.real));
    }

    #[test]
    fn test_wsd_open_root_refusals_name_the_root() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let root = tmp.path().join("root");
        std::fs::create_dir_all(&root).unwrap_or_else(|e| panic!("{e}"));
        let (policy, _) = handle(&root, false);
        let e = open_root(&policy, &Value::from("nope"), false)
            .err()
            .unwrap_or_else(|| panic!("found"));
        assert_eq!(
            e.message,
            "there is no model root 'nope' in this node's policy (/x/policy.json)"
        );
        let e = open_root(&policy, &Value::Null, false)
            .err()
            .unwrap_or_else(|| panic!("found"));
        assert_eq!(
            e.message,
            "there is no model root 'None' in this node's policy (/x/policy.json)"
        );
        let e = open_root(&policy, &Value::from("t"), true)
            .err()
            .unwrap_or_else(|| panic!("writable"));
        assert_eq!(e.code, POLICY_DENIED);
        assert!(
            e.message
                .starts_with("model root 't' is read-only in this node's policy")
        );
        std::fs::remove_dir(&root).unwrap_or_else(|e| panic!("{e}"));
        let e = open_root(&policy, &Value::from("t"), false)
            .err()
            .unwrap_or_else(|| panic!("missing"));
        assert_eq!(e.code, NOT_FOUND);
        assert!(
            e.message
                .ends_with("does not exist on this machine (not mounted?)")
        );
    }

    #[test]
    fn test_wsd_open_dir_create_refuses_links_out_and_files_in_the_way() {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let root = tmp.path().join("root");
        std::fs::create_dir_all(root.join("d")).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(root.join("file"), "x").unwrap_or_else(|e| panic!("{e}"));
        std::os::unix::fs::symlink("..", root.join("up")).unwrap_or_else(|e| panic!("{e}"));
        std::os::unix::fs::symlink("d", root.join("in")).unwrap_or_else(|e| panic!("{e}"));
        let (_, h) = handle(&root, true);
        let parts = |v: &[&str]| v.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        for (p, frag) in [
            (&["up", "new"][..], "'up' is a link"),
            (&["file", "new"][..], "'file' is a link"),
        ] {
            match open_dir(&h, &parts(p), true) {
                Err(Fail::Op(e)) => assert!(
                    e.code == POLICY_DENIED && e.message.starts_with(frag),
                    "{e:?}"
                ),
                other => panic!("{p:?}: {:?}", other.map(|_| "fd")),
            }
        }
        assert!(!tmp.path().join("new").exists());
        // a link that stays inside the root is followed; missing directories are made 0755
        let fd = open_dir(&h, &parts(&["in", "made", "deeper"]), true)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let real = py::realpath(&root.join("d/made/deeper"));
        assert_eq!(fd_path(fd.as_fd()), Some(real));
    }
}
