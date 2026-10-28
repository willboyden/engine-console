//! Python: node/src/ecnode/fsops.py.
//!
//! File operations inside policy model roots (docs/FLEET-PROTOCOL.md §4): fs.stat, fs.write, fs.read, fs.symlink,
//! fs.put_small and fs.delete.
//!
//! Every path goes through confine::split_rel + confine::open_dir: relative, no "..", parent realpath inside the
//! root, final component opened with O_NOFOLLOW relative to the checked directory fd. Uploads land in
//! `<rel>.incomplete` and are renamed into place only after the size (and, when given, the sha256 of the WHOLE
//! file) matched.
//!
//! Error mapping follows Python's try/except structure: an OSError inside an operation's guarded section becomes
//! the op's own error (io_error, no_space); one Python lets escape becomes `internal` naming its exception class.

use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex, MutexGuard};

use ec_proto::{
    BAD_ARGS, CONFLICT, HASH_MISMATCH, IN_USE, IO_ERROR, NO_SPACE, NOT_FOUND, Obj, POLICY_DENIED,
    SIZE_MISMATCH,
};
use nix::errno::Errno;
use nix::fcntl::{Flock, FlockArg, OFlag};
use nix::sys::stat::{FileStat, Mode};
use nix::unistd::UnlinkatFlags;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::confine::{
    Fail, dir_flags, fd_path, inside, lstat_at, lstat_at_os, open_dir, open_root,
    resolve_link_inside, secret_inside, split_rel, split_rel_for_write, strerror,
};
use crate::errors::{HandlerError, OpError, Stream};
use crate::policy::Policy;
use crate::util::{is_secret_name, py};

pub const PARTIAL: &str = ".incomplete";
/// never fill a disk below 1 GiB free
pub const MIN_FREE: i128 = 1 << 30;
pub const CHUNK: usize = 512 * 1024;
pub const PUT_SMALL_MAX: usize = 96 * 1024;
pub const MAX_FILE: i64 = 1 << 50;
pub const MAX_DELETE_DEPTH: usize = 64;

static SHA: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"\A[0-9a-fA-F]{64}\z").unwrap_or_else(|_| std::process::abort())
});
static BAD_TARGET: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"[\x00-\x1f\x7f\\]").unwrap_or_else(|_| std::process::abort())
});

fn op(code: &str, msg: impl Into<String>) -> OpError {
    OpError::new(code, msg)
}

/// An error inside an operation's guarded section: a handler error, or an OSError the section maps itself.
enum E {
    H(HandlerError),
    Os(i32),
}

impl From<HandlerError> for E {
    fn from(e: HandlerError) -> Self {
        E::H(e)
    }
}

impl From<OpError> for E {
    fn from(e: OpError) -> Self {
        E::H(HandlerError::Op(e))
    }
}

impl From<Errno> for E {
    fn from(e: Errno) -> Self {
        E::Os(e as i32)
    }
}

impl From<std::io::Error> for E {
    fn from(e: std::io::Error) -> Self {
        E::Os(py::errno_of(&e))
    }
}

impl From<crate::errors::Cancelled> for E {
    fn from(_: crate::errors::Cancelled) -> Self {
        E::H(HandlerError::Cancelled)
    }
}

fn is_reg(st: &FileStat) -> bool {
    st.st_mode & nix::libc::S_IFMT == nix::libc::S_IFREG
}

fn is_dir(st: &FileStat) -> bool {
    st.st_mode & nix::libc::S_IFMT == nix::libc::S_IFDIR
}

fn is_lnk(st: &FileStat) -> bool {
    st.st_mode & nix::libc::S_IFMT == nix::libc::S_IFLNK
}

/// A FIFO, socket or device node. No fs op reads, writes, replaces or uploads over one (node hardening: opening a
/// FIFO blocks, opening a device can have side effects, and replacing either is never what an upload means).
fn is_special(st: &FileStat) -> bool {
    matches!(
        st.st_mode & nix::libc::S_IFMT,
        nix::libc::S_IFIFO | nix::libc::S_IFSOCK | nix::libc::S_IFCHR | nix::libc::S_IFBLK
    )
}

fn special_refused(what: &str) -> OpError {
    op(
        CONFLICT,
        format!("{what} is a FIFO, socket or device file; refusing to replace it"),
    )
}

fn size_of(st: &FileStat) -> i64 {
    st.st_size
}

fn arg<'a>(args: &'a Obj, k: &str) -> &'a Value {
    args.get(k).unwrap_or(&Value::Null)
}

/// `isinstance(v, int)` (True and False count, as in Python).
fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) if !n.is_f64() => Some(n.as_i64().unwrap_or(i64::MAX)),
        Value::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

/// `args.get(k) or 0` for an integer argument; anything else is the TypeError Python would hit when comparing it.
fn int_or_zero(args: &Obj, k: &str) -> Result<i64, HandlerError> {
    let v = arg(args, k);
    if !py::truthy(v) {
        return Ok(0);
    }
    py_int(v).ok_or_else(|| HandlerError::internal("TypeError"))
}

fn rel_repr(rel: &str) -> String {
    py::repr_take(rel, 300)
}

fn name_ok_for_partial(name: &str) -> Result<(), OpError> {
    if name.len() + PARTIAL.len() > 255 {
        return Err(op(
            BAD_ARGS,
            "the file name is too long to hold an upload in progress (max 243 bytes)",
        ));
    }
    Ok(())
}

fn with_partial(name: &str) -> String {
    format!("{name}{PARTIAL}")
}

// ── fs.stat ──────────────────────────────────────────────────────────────────────────────────────────────
pub fn fs_stat(policy: &Policy, args: &Obj) -> Result<Obj, HandlerError> {
    let h = open_root(policy, arg(args, "root"), false)?;
    let paths: Vec<Value> = arg(args, "paths")
        .as_array()
        .map(|a| a.iter().take(256).cloned().collect())
        .unwrap_or_default();
    let mut items: Vec<Value> = Vec::new();
    for rel in paths {
        let parts = split_rel(rel.as_str(), "path")?;
        let mut item =
            json!({"rel": rel, "exists": false, "is_dir": false, "size": null, "partial": 0});
        let dfd = match open_dir(&h, &parts[..parts.len() - 1], false) {
            Ok(fd) => fd,
            Err(Fail::Op(e)) if e.code == NOT_FOUND || e.code == BAD_ARGS => {
                items.push(item);
                continue;
            }
            Err(f) => return Err(f.into()),
        };
        let name = &parts[parts.len() - 1];
        let mut st = lstat_at(dfd.as_fd(), name)?;
        if st.as_ref().is_some_and(is_lnk) {
            st = match resolve_link_inside(dfd.as_fd(), name, &h) {
                Some(real) if std::fs::metadata(&real).is_ok() => Some(
                    nix::sys::stat::stat(&real)
                        .map_err(|e| HandlerError::internal(py::os_error_name(e as i32)))?,
                ),
                _ => None,
            };
        }
        if let Some(st) = &st {
            item["exists"] = json!(true);
            item["is_dir"] = json!(is_dir(st));
            item["size"] = if is_reg(st) {
                json!(size_of(st))
            } else {
                Value::Null
            };
        }
        if name.len() + PARTIAL.len() <= 255
            && let Some(pst) = lstat_at(dfd.as_fd(), &with_partial(name))?
            && is_reg(&pst)
        {
            item["partial"] = json!(size_of(&pst));
        }
        items.push(item);
    }
    let mut out = Obj::new();
    out.insert("items".into(), Value::Array(items));
    Ok(out)
}

// ── fs.write ─────────────────────────────────────────────────────────────────────────────────────────────
/// One upload per file at a time within this node (flock on the partial covers other processes).
#[derive(Debug, Default)]
pub struct WriteLocks {
    held: Mutex<HashSet<Vec<u8>>>,
}

/// A held write lock; released when dropped.
pub struct Held<'a> {
    locks: &'a WriteLocks,
    key: Vec<u8>,
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        self.locks.lock().remove(&self.key);
    }
}

impl WriteLocks {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, HashSet<Vec<u8>>> {
        crate::util::lock(&self.held)
    }

    pub fn held(&self) -> usize {
        self.lock().len()
    }

    /// `WriteLocks.hold(key)`: conflict while another upload holds the same key.
    pub fn hold(&self, key: &[u8]) -> Result<Held<'_>, OpError> {
        if !self.lock().insert(key.to_vec()) {
            return Err(op(CONFLICT, "another upload to that file is in progress"));
        }
        Ok(Held {
            locks: self,
            key: key.to_vec(),
        })
    }
}

fn unlink_quiet(dfd: BorrowedFd<'_>, name: &OsStr) {
    let _ = nix::unistd::unlinkat(dfd, name, UnlinkatFlags::NoRemoveDir);
}

/// The real `os.fstatvfs`: (f_bavail, f_frsize).
pub fn os_fstatvfs(fd: BorrowedFd<'_>) -> nix::Result<(u64, u64)> {
    let v = nix::sys::statvfs::fstatvfs(fd)?;
    Ok((v.blocks_available(), v.fragment_size()))
}

pub fn fs_write(
    policy: &Policy,
    args: &Obj,
    stream: &dyn Stream,
    locks: &WriteLocks,
) -> Result<Obj, HandlerError> {
    fs_write_with(policy, args, stream, locks, &os_fstatvfs)
}

/// fs_write with the free-space probe injectable (tests give it a nearly full disk).
pub fn fs_write_with(
    policy: &Policy,
    args: &Obj,
    stream: &dyn Stream,
    locks: &WriteLocks,
    fstatvfs: &dyn Fn(BorrowedFd<'_>) -> nix::Result<(u64, u64)>,
) -> Result<Obj, HandlerError> {
    let h = open_root(policy, arg(args, "root"), true)?;
    let rel_v = arg(args, "rel");
    let parts = split_rel_for_write(rel_v.as_str(), "path")?;
    let rel = rel_v.as_str().unwrap_or_default();
    let name = parts[parts.len() - 1].clone();
    name_ok_for_partial(&name)?;
    let size = match py_int(arg(args, "size")) {
        Some(s) if (0..=MAX_FILE).contains(&s) => s,
        _ => return Err(op(BAD_ARGS, "size must be the whole file's size in bytes").into()),
    };
    let want = match arg(args, "sha256") {
        Value::Null => None,
        Value::String(s) if SHA.is_match(s) => Some(s.to_lowercase()),
        Value::String(_) => return Err(op(BAD_ARGS, "sha256 must be 64 hex digits").into()),
        _ => return Err(HandlerError::internal("TypeError")),
    };
    let offset = int_or_zero(args, "offset")?;
    if !(0 <= offset && offset <= size) {
        return Err(op(
            BAD_ARGS,
            format!("offset {offset} is outside the file (0-{size})"),
        )
        .into());
    }
    let mut key = h.real.as_os_str().as_bytes().to_vec();
    key.push(0);
    key.extend_from_slice(rel.as_bytes());
    let _held = locks.hold(&key)?;
    let dfd = open_dir(&h, &parts[..parts.len() - 1], true)?;
    let partial = with_partial(&name);
    let res = (|| -> Result<Obj, E> {
        let st = lstat_at(dfd.as_fd(), &name)?;
        if st.as_ref().is_some_and(is_dir) {
            return Err(op(CONFLICT, format!("{} is a directory", rel_repr(rel))).into());
        }
        if st.as_ref().is_some_and(is_special) {
            return Err(special_refused(&rel_repr(rel)).into());
        }
        // a special file in the partial's place is refused before it is opened (the fstat below re-checks)
        if lstat_at(dfd.as_fd(), &partial)?
            .as_ref()
            .is_some_and(is_special)
        {
            return Err(op(CONFLICT, "the upload's partial file is not a regular file").into());
        }
        let pfd = match nix::fcntl::openat(
            dfd.as_fd(),
            partial.as_str(),
            OFlag::O_RDWR
                | OFlag::O_CREAT
                | OFlag::O_NOFOLLOW
                | OFlag::O_CLOEXEC
                | OFlag::O_NONBLOCK,
            Mode::from_bits_truncate(0o644),
        ) {
            Ok(fd) => fd,
            Err(Errno::ELOOP) => {
                return Err(op(
                    POLICY_DENIED,
                    "the upload's partial file is a symlink; refusing",
                )
                .into());
            }
            Err(e) => return Err(e.into()),
        };
        let file = match Flock::lock(File::from(pfd), FlockArg::LockExclusiveNonblock) {
            Ok(f) => f,
            Err((_, Errno::EWOULDBLOCK)) => {
                return Err(op(CONFLICT, "another process is writing that file").into());
            }
            Err((_, e)) => return Err(e.into()),
        };
        let pst = nix::sys::stat::fstat(file.as_fd())?;
        if !is_reg(&pst) {
            return Err(op(CONFLICT, "the upload's partial file is not a regular file").into());
        }
        let current = size_of(&pst);
        if offset == 0 {
            nix::unistd::ftruncate(file.as_fd(), 0)?;
        } else if offset != current {
            return Err(op(
                CONFLICT,
                format!(
                    "resume offset {offset} does not match the partial file ({current} bytes): resume from \
                     {current}, or start at 0"
                ),
            )
            .into());
        }
        let (bavail, frsize) = fstatvfs(file.as_fd())?;
        let free = i128::from(bavail) * i128::from(frsize);
        let more = i128::from(size - offset);
        if free - more < MIN_FREE {
            return Err(op(
                NO_SPACE,
                format!(
                    "writing {more} more bytes would leave less than 1 GiB free on that disk ({free} bytes free)"
                ),
            )
            .into());
        }
        let mut hasher = (want.is_some() || offset == 0).then(Sha256::new);
        if let Some(hs) = hasher.as_mut()
            && offset > 0
        {
            // sha256 is of the WHOLE file: feed the existing part first
            let mut pos: i64 = 0;
            let mut buf = vec![0u8; 4 << 20];
            while pos < offset {
                stream.check()?;
                let want_n = usize::try_from((offset - pos).min(4 << 20)).unwrap_or(0);
                let n = file.read_at(&mut buf[..want_n], u64::try_from(pos).unwrap_or(0))?;
                if n == 0 {
                    return Err(
                        op(IO_ERROR, "the partial file shrank while it was being read").into(),
                    );
                }
                hs.update(&buf[..n]);
                pos += i64::try_from(n).unwrap_or(0);
            }
        }
        (&*file).seek(SeekFrom::Start(u64::try_from(offset).unwrap_or(0)))?;
        let mut total = offset;
        let mut overflow = false;
        loop {
            let chunk = stream.read_upload()?;
            if chunk.is_empty() {
                break;
            }
            let n = i64::try_from(chunk.len()).unwrap_or(i64::MAX);
            if total + n > size {
                overflow = true;
                break;
            }
            (&*file).write_all(&chunk)?;
            if let Some(hs) = hasher.as_mut() {
                hs.update(&chunk);
            }
            total += n;
        }
        if overflow || total != size {
            unlink_quiet(dfd.as_fd(), OsStr::new(&partial));
            let got = if overflow {
                "more than".to_string()
            } else {
                format!("{total} of")
            };
            return Err(op(
                SIZE_MISMATCH,
                format!(
                    "received {got} {size} bytes for {}; the partial file was deleted",
                    rel_repr(rel)
                ),
            )
            .into());
        }
        file.sync_all()?;
        let digest = hasher.map(|hs| hex::encode(hs.finalize()));
        if let Some(w) = &want
            && digest.as_ref() != Some(w)
        {
            unlink_quiet(dfd.as_fd(), OsStr::new(&partial));
            return Err(op(
                HASH_MISMATCH,
                format!(
                    "sha256 of {} does not match; the partial file was deleted",
                    rel_repr(rel)
                ),
            )
            .into());
        }
        nix::fcntl::renameat(dfd.as_fd(), partial.as_str(), dfd.as_fd(), name.as_str())?;
        let _ = nix::unistd::fsync(dfd.as_fd());
        let mut out = Obj::new();
        out.insert("size".into(), json!(size));
        out.insert("sha256".into(), json!(digest));
        Ok(out)
    })();
    match res {
        Ok(o) => Ok(o),
        Err(E::H(e)) => Err(e),
        Err(E::Os(n)) if n == Errno::ENOSPC as i32 || n == Errno::EDQUOT as i32 => Err(op(
            NO_SPACE,
            "the disk filled up during the upload; the partial file was kept for a resume",
        )
        .into()),
        Err(E::Os(n)) => Err(op(
            IO_ERROR,
            format!("writing {} failed: {}", rel_repr(rel), strerror(n)),
        )
        .into()),
    }
}

// ── fs.read ──────────────────────────────────────────────────────────────────────────────────────────────
/// Stream <root>/<rel> from `offset` (for `length` bytes, default: to the end). Result size = bytes sent.
pub fn fs_read(policy: &Policy, args: &Obj, stream: &dyn Stream) -> Result<Obj, HandlerError> {
    let h = open_root(policy, arg(args, "root"), false)?;
    let rel_v = arg(args, "rel");
    let parts = split_rel(rel_v.as_str(), "path")?;
    let rel = rel_v.as_str().unwrap_or_default();
    if parts.iter().any(|p| is_secret_name(p)) {
        return Err(op(
            POLICY_DENIED,
            "refusing to read a file that looks like a key, token or credential",
        )
        .into());
    }
    let fd: OwnedFd = {
        let dfd = open_dir(&h, &parts[..parts.len() - 1], false)?;
        if let Some(where_) = fd_path(dfd.as_fd())
            && secret_inside(&where_, &h.real)
        {
            return Err(op(
                POLICY_DENIED,
                "refusing to read from a directory that looks like a key or credential store",
            )
            .into());
        }
        // a FIFO, socket, device or directory is refused before it is opened (Python opens it, then refuses it with
        // the same message after fstat; opening a device can have side effects). A symlink still reaches the open,
        // whose O_NOFOLLOW gives the "is a symbolic link" answer.
        if let Some(st) = lstat_at(dfd.as_fd(), &parts[parts.len() - 1])?
            && !is_reg(&st)
            && !is_lnk(&st)
        {
            return Err(op(BAD_ARGS, format!("{} is not a regular file", rel_repr(rel))).into());
        }
        match nix::fcntl::openat(
            dfd.as_fd(),
            parts[parts.len() - 1].as_str(),
            OFlag::O_RDONLY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC | OFlag::O_NONBLOCK,
            Mode::empty(),
        ) {
            Ok(fd) => fd,
            Err(Errno::ENOENT) => {
                return Err(op(
                    NOT_FOUND,
                    format!(
                        "{} does not exist in model root {}",
                        rel_repr(rel),
                        py::repr(&h.root.id)
                    ),
                )
                .into());
            }
            Err(Errno::ELOOP) => {
                return Err(op(
                    BAD_ARGS,
                    format!(
                        "{} is a symbolic link; read the file it points to (in an HF cache: the blob)",
                        rel_repr(rel)
                    ),
                )
                .into());
            }
            Err(e) => {
                return Err(op(
                    IO_ERROR,
                    format!("cannot open {}: {}", rel_repr(rel), strerror(e as i32)),
                )
                .into());
            }
        }
    };
    let file = File::from(fd);
    let res = (|| -> Result<Obj, E> {
        let st = nix::sys::stat::fstat(file.as_fd())?;
        if !is_reg(&st) {
            return Err(op(BAD_ARGS, format!("{} is not a regular file", rel_repr(rel))).into());
        }
        let fsize = size_of(&st);
        let offset = int_or_zero(args, "offset")?;
        if !(0 <= offset && offset <= fsize) {
            return Err(op(
                BAD_ARGS,
                format!("offset {offset} is outside the file (0-{fsize})"),
            )
            .into());
        }
        let length = match arg(args, "length") {
            Value::Null => None,
            v => Some(py_int(v).ok_or_else(|| HandlerError::internal("TypeError"))?),
        };
        if length.is_some_and(|l| l < 0) {
            return Err(op(BAD_ARGS, "length must not be negative").into());
        }
        let remaining = match length {
            None => fsize - offset,
            Some(l) => l.min(fsize - offset),
        };
        let (mut sent, mut pos) = (0i64, offset);
        let mut buf = vec![0u8; CHUNK];
        while sent < remaining {
            stream.check()?;
            let want = usize::try_from((remaining - sent).min(i64::try_from(CHUNK).unwrap_or(0)))
                .unwrap_or(0);
            let n = file.read_at(&mut buf[..want], u64::try_from(pos).unwrap_or(0))?;
            if n == 0 {
                break; // the file shrank while it was being read
            }
            stream.send_data(&buf[..n])?;
            let n = i64::try_from(n).unwrap_or(0);
            sent += n;
            pos += n;
        }
        let mut out = Obj::new();
        out.insert("size".into(), json!(sent));
        Ok(out)
    })();
    match res {
        Ok(o) => Ok(o),
        Err(E::H(e)) => Err(e),
        Err(E::Os(n)) => Err(op(
            IO_ERROR,
            format!("reading {} failed: {}", rel_repr(rel), strerror(n)),
        )
        .into()),
    }
}

// ── fs.symlink / fs.put_small ────────────────────────────────────────────────────────────────────────────
/// `secrets.token_hex(n)`
fn token_hex(n: usize) -> String {
    let mut b = vec![0u8; n];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut b))
        .is_err()
    {
        // never reached on Linux; the name only has to be unlikely to exist already
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        for (i, x) in b.iter_mut().enumerate() {
            *x = u8::try_from((t >> (8 * (i % 16))) & 0xff).unwrap_or(0)
                ^ u8::try_from(i).unwrap_or(0);
        }
    }
    hex::encode(b)
}

/// Create a temp entry with `make(tmp)` in dfd, then rename it over `name` (a file or link, never a dir).
fn replace_into(
    dfd: BorrowedFd<'_>,
    name: &str,
    make: &dyn Fn(&str) -> Result<(), E>,
) -> Result<(), HandlerError> {
    let st = lstat_at(dfd, name)?;
    if st.as_ref().is_some_and(is_dir) {
        return Err(op(
            CONFLICT,
            format!("{} is a directory", py::repr_take(name, 200)),
        )
        .into());
    }
    if st.as_ref().is_some_and(is_special) {
        return Err(special_refused(&py::repr_take(name, 200)).into());
    }
    let tmp = format!(".{}.ecnode-{}", py::take(name, 200), token_hex(6));
    let res = make(&tmp)
        .and_then(|()| nix::fcntl::renameat(dfd, tmp.as_str(), dfd, name).map_err(E::from));
    match res {
        Ok(()) => {}
        Err(E::H(e)) => return Err(e),
        Err(E::Os(n)) => {
            unlink_quiet(dfd, OsStr::new(&tmp));
            if n == Errno::ENOSPC as i32 || n == Errno::EDQUOT as i32 {
                return Err(op(NO_SPACE, "the disk is full").into());
            }
            return Err(op(
                IO_ERROR,
                format!("cannot write {}: {}", py::repr_take(name, 200), strerror(n)),
            )
            .into());
        }
    }
    let _ = nix::unistd::fsync(dfd);
    Ok(())
}

/// <root>/<link> -> relative `target`; both must resolve inside the same root (HF snapshot links).
pub fn fs_symlink(policy: &Policy, args: &Obj) -> Result<Obj, HandlerError> {
    let h = open_root(policy, arg(args, "root"), true)?;
    let parts = split_rel_for_write(arg(args, "link").as_str(), "link")?;
    let target = match arg(args, "target") {
        Value::String(t)
            if !t.is_empty()
                && !t.starts_with('/')
                && !BAD_TARGET.is_match(t)
                && !t.split('/').any(|s| s.is_empty() || s == ".") =>
        {
            t.clone()
        }
        _ => {
            return Err(op(
                BAD_ARGS,
                "the link target must be a relative path like ../../blobs/<hash>",
            )
            .into());
        }
    };
    let segs: Vec<&str> = target.split('/').collect();
    let ups = segs.iter().take_while(|s| **s == "..").count();
    if segs[ups..].contains(&"..") {
        // `..` only as leading segments (HF links are ../../blobs/<hash>). After a normal segment the kernel applies
        // it to wherever that segment really leads, which a lexical check cannot see: `y -> sub/..` then
        // `z -> y/..` would point z at the root's parent.
        return Err(op(
            POLICY_DENIED,
            "the link target may use '..' only at its start, like ../../blobs/<hash>",
        )
        .into());
    }
    let parent = &parts[..parts.len() - 1];
    let dfd = open_dir(&h, parent, true)?;
    let base = fd_path(dfd.as_fd()).unwrap_or_else(|| h.path(parent));
    // realpath of the joined string, never normpath first: the kernel resolves `..` physically
    if !inside(&py::realpath(&py::join(&base, &[target.as_str()])), &h.real) {
        return Err(op(
            POLICY_DENIED,
            "the link target leads outside the model root",
        )
        .into());
    }
    let d = dfd.as_fd();
    replace_into(d, &parts[parts.len() - 1], &|tmp: &str| {
        nix::unistd::symlinkat(target.as_str(), d, tmp).map_err(E::from)
    })?;
    Ok(Obj::new())
}

fn b64_value(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// `base64.b64decode(s, validate=True)` (Python 3.12: `binascii.a2b_base64(s, strict_mode=True)`); None where
/// Python raises binascii.Error or ValueError.
pub fn b64decode_strict(s: &str) -> Option<Vec<u8>> {
    if !s.is_ascii() {
        return None; // "string argument should contain only ASCII characters"
    }
    let data = s.as_bytes();
    let mut out = Vec::with_capacity(data.len() / 4 * 3);
    let (mut quad_pos, mut pads, mut leftchar) = (0u8, 0u8, 0u32);
    let mut padding_started = false;
    for (i, &c) in data.iter().enumerate() {
        if c == b'=' {
            padding_started = true;
            if quad_pos == 0 {
                return None; // leading / excess padding
            }
            if quad_pos >= 2 {
                pads += 1;
                if quad_pos + pads >= 4 {
                    if i + 1 < data.len() {
                        return None; // excess data after padding
                    }
                    return Some(out);
                }
            }
            continue;
        }
        let v = u32::from(b64_value(c)?); // only base64 data is allowed
        if padding_started {
            return None; // discontinuous padding
        }
        pads = 0;
        match quad_pos {
            0 => {
                leftchar = v;
                quad_pos = 1;
            }
            1 => {
                leftchar = (leftchar << 6) | v;
                out.push(u8::try_from(leftchar >> 4).unwrap_or(0));
                leftchar &= 0x0f;
                quad_pos = 2;
            }
            2 => {
                leftchar = (leftchar << 6) | v;
                out.push(u8::try_from(leftchar >> 2).unwrap_or(0));
                leftchar &= 0x03;
                quad_pos = 3;
            }
            _ => {
                out.push(u8::try_from(((leftchar << 6) | v) & 0xff).unwrap_or(0));
                leftchar = 0;
                quad_pos = 0;
            }
        }
    }
    if quad_pos != 0 {
        return None; // 1 more than a multiple of 4, or incorrect padding
    }
    Some(out)
}

/// Write at most 96 KiB atomically (HF refs/main, small metadata).
pub fn fs_put_small(policy: &Policy, args: &Obj) -> Result<Obj, HandlerError> {
    let h = open_root(policy, arg(args, "root"), true)?;
    let parts = split_rel_for_write(arg(args, "rel").as_str(), "path")?;
    let content = match arg(args, "content_b64") {
        Value::String(s) => {
            b64decode_strict(s).ok_or_else(|| op(BAD_ARGS, "content_b64 is not valid base64"))?
        }
        _ => return Err(HandlerError::internal("TypeError")),
    };
    if content.len() > PUT_SMALL_MAX {
        return Err(op(
            BAD_ARGS,
            format!("fs.put_small takes at most {PUT_SMALL_MAX} bytes; use fs.write"),
        )
        .into());
    }
    let dfd = open_dir(&h, &parts[..parts.len() - 1], true)?;
    let d = dfd.as_fd();
    replace_into(d, &parts[parts.len() - 1], &|tmp: &str| {
        let fd = nix::fcntl::openat(
            d,
            tmp,
            OFlag::O_WRONLY | OFlag::O_CREAT | OFlag::O_EXCL | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::from_bits_truncate(0o644),
        )?;
        let mut f = File::from(fd);
        f.write_all(&content)?;
        f.sync_all()?;
        Ok(())
    })?;
    Ok(Obj::new())
}

// ── fs.delete ────────────────────────────────────────────────────────────────────────────────────────────
/// Bytes a regular file gives back, each inode once (HF snapshot links and hard links share blobs).
fn count(st: &FileStat, seen: &mut HashSet<(u64, u64)>) -> i64 {
    if !is_reg(st) || !seen.insert((st.st_dev, st.st_ino)) {
        return 0;
    }
    size_of(st)
}

/// `[e.name for e in os.scandir(dirfd)]`
fn names_in(dirfd: BorrowedFd<'_>) -> Result<Vec<OsString>, E> {
    let mut dir = nix::dir::Dir::from_fd(nix::unistd::dup(dirfd)?)?;
    let mut names = Vec::new();
    for entry in dir.iter() {
        let entry = entry?;
        let n = entry.file_name().to_bytes();
        if n != b"." && n != b".." {
            names.push(OsString::from_vec(n.to_vec()));
        }
    }
    Ok(names)
}

/// Remove everything below the already-open directory `dirfd`; return the bytes freed. Only fds and names relative
/// to them: no path string is ever resolved again, so a component swapped for a link while this runs is opened
/// with O_NOFOLLOW (and refused), never followed. Links themselves are unlinked.
fn empty_dir_at(
    dirfd: BorrowedFd<'_>,
    seen: &mut HashSet<(u64, u64)>,
    depth: usize,
) -> Result<i64, E> {
    if depth > MAX_DELETE_DEPTH {
        return Err(op(
            BAD_ARGS,
            format!(
                "the tree is more than {MAX_DELETE_DEPTH} directories deep; the delete stopped part-way (what was \
                 removed stays removed)"
            ),
        )
        .into());
    }
    let mut freed = 0;
    for name in names_in(dirfd)? {
        let Some(st) = lstat_at_os(dirfd, &name)? else {
            continue;
        };
        if is_dir(&st) {
            // O_NOFOLLOW: a directory swapped for a link fails here
            let sub = nix::fcntl::openat(dirfd, name.as_os_str(), dir_flags(), Mode::empty())?;
            freed += empty_dir_at(sub.as_fd(), seen, depth + 1)?;
            drop(sub);
            nix::unistd::unlinkat(dirfd, name.as_os_str(), UnlinkatFlags::RemoveDir)?;
        } else {
            freed += count(&st, seen);
            nix::unistd::unlinkat(dirfd, name.as_os_str(), UnlinkatFlags::NoRemoveDir)?;
        }
    }
    Ok(freed)
}

/// Delete one file or directory tree (policy allow_delete, writable root, nothing running uses it).
///
/// The parent directory and (for a tree) the target are opened once and pinned by fd; the checks run on the fd's
/// real location and the removal goes through those fds, so a path re-pointed between the check and the delete
/// (the in-use check runs docker, which takes long enough to race) cannot move it outside the root.
pub fn fs_delete(
    policy: &Policy,
    args: &Obj,
    running_mounts: &dyn Fn() -> Result<Vec<String>, OpError>,
) -> Result<Obj, HandlerError> {
    if !policy.allow_delete {
        return Err(op(
            POLICY_DENIED,
            format!(
                "this node's policy does not allow deleting models (allow_delete in {})",
                policy.where_()
            ),
        )
        .into());
    }
    let h = open_root(policy, arg(args, "root"), true)?;
    let rel_v = arg(args, "rel");
    let parts = split_rel(rel_v.as_str(), "path")?;
    let rel = rel_v.as_str().unwrap_or_default();
    let dfd = open_dir(&h, &parts[..parts.len() - 1], false)?;
    let name = &parts[parts.len() - 1];
    let Some(st) = lstat_at(dfd.as_fd(), name)? else {
        return Err(op(
            NOT_FOUND,
            format!(
                "{} does not exist in model root {}",
                rel_repr(rel),
                py::repr(&h.root.id)
            ),
        )
        .into());
    };
    let parent = match fd_path(dfd.as_fd()) {
        Some(p) if inside(&p, &h.real) => p,
        _ => {
            return Err(op(
                POLICY_DENIED,
                "refusing to delete: cannot confirm where that path really is",
            )
            .into());
        }
    };
    let mut tfd: Option<OwnedFd> = None;
    let real: PathBuf = if is_dir(&st) {
        match nix::fcntl::openat(dfd.as_fd(), name.as_str(), dir_flags(), Mode::empty()) {
            Ok(fd) => {
                let r = fd_path(fd.as_fd()).unwrap_or_default();
                tfd = Some(fd);
                r
            }
            Err(e) => {
                return Err(op(
                    CONFLICT,
                    format!(
                        "{} changed while being deleted: {}",
                        rel_repr(rel),
                        strerror(e as i32)
                    ),
                )
                .into());
            }
        }
    } else {
        // a file or a link: removed by name in its parent
        py::join(&parent, &[name.as_str()])
    };
    if real.as_os_str().is_empty() || !inside(&real, &h.real) || real == h.real {
        return Err(op(
            POLICY_DENIED,
            "refusing to delete the model root itself or anything outside it",
        )
        .into());
    }
    for other in &policy.model_roots {
        let oreal = py::realpath(Path::new(&other.path));
        if other.id != h.root.id && inside(&oreal, &real) {
            return Err(op(
                POLICY_DENIED,
                format!(
                    "refusing to delete {}: it holds model root {}",
                    rel_repr(rel),
                    py::repr(&other.id)
                ),
            )
            .into());
        }
    }
    let mounts = running_mounts()?;
    let users = mounts
        .iter()
        .any(|m| inside(&py::realpath(Path::new(m)), &real) || inside(Path::new(m), &real));
    if users {
        return Err(op(
            IN_USE,
            format!(
                "a running container mounts {}; stop it before deleting",
                rel_repr(rel)
            ),
        )
        .into());
    }
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    let res = (|| -> Result<i64, E> {
        if let Some(t) = &tfd {
            if fd_path(t.as_fd()).unwrap_or_default() != real {
                return Err(op(
                    CONFLICT,
                    format!(
                        "{} moved while being deleted; nothing was removed",
                        rel_repr(rel)
                    ),
                )
                .into());
            }
            let freed = empty_dir_at(t.as_fd(), &mut seen, 0)?;
            nix::unistd::unlinkat(dfd.as_fd(), name.as_str(), UnlinkatFlags::RemoveDir)?;
            Ok(freed)
        } else {
            let freed = count(&st, &mut seen);
            nix::unistd::unlinkat(dfd.as_fd(), name.as_str(), UnlinkatFlags::NoRemoveDir)?;
            Ok(freed)
        }
    })();
    match res {
        Ok(freed) => {
            let mut out = Obj::new();
            out.insert("freed".into(), json!(freed));
            Ok(out)
        }
        Err(E::H(e)) => Err(e),
        Err(E::Os(n)) => Err(op(
            IO_ERROR,
            format!(
                "deleting {} failed part-way: {}",
                rel_repr(rel),
                strerror(n)
            ),
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64_strict_matches_python_3_12() {
        // vectors from Python's base64.b64decode(s, validate=True)
        let ok: &[(&str, &[u8])] = &[
            ("YWJj", b"abc"),
            ("YWI=", b"ab"),
            ("YW==", b"a"),
            ("YR==", b"a"),
            ("", b""),
        ];
        for (s, want) in ok {
            assert_eq!(b64decode_strict(s).as_deref(), Some(*want), "{s}");
        }
        for s in [
            "YWI",
            "Y===",
            "YWJj==",
            "YQ==YQ==",
            "=YWJj",
            "YWJjZ",
            "YW Jj",
            "YQ=",
            "%%%",
            "YWJj\u{e9}",
        ] {
            assert_eq!(b64decode_strict(s), None, "{s}");
        }
    }

    #[test]
    fn write_locks_release_on_drop() {
        let l = WriteLocks::new();
        let a = l.hold(b"k");
        assert!(a.is_ok());
        assert_eq!(
            l.hold(b"k").map(|_| ()).map_err(|e| e.code),
            Err(CONFLICT.to_string())
        );
        drop(a);
        assert!(l.hold(b"k").is_ok());
        assert_eq!(l.held(), 0);
    }
}
