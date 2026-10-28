//! Reading the hub's own files without hanging or ballooning.
//!
//! Python opens these files with `open()` / `Path.read_text()`: a FIFO planted at a settings path hangs the request
//! (or start-up) forever, a device or a huge file is read whole. Every file the hub reads goes through
//! [`open_regular`] instead: `O_NONBLOCK` (opening a FIFO returns at once), `O_NOFOLLOW` where the file is the hub's
//! own and has no reason to be a symlink, then `fstat` on the descriptor (never the path again): anything but a
//! regular file is refused. [`read_capped`] then reads at most a fixed number of bytes.

use std::fs::File;
use std::io::{self, Read};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// Whether a symlink in the last component of the path is followed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Follow {
    /// Python follows it and a user may rely on that (an env file or token file linked from elsewhere).
    Yes,
    /// The hub's own file (secrets, frontend assets after canonicalisation): a symlink there is refused (ELOOP).
    No,
}

/// The error of a path that exists but is not a regular file (a FIFO, a socket, a device, a directory).
pub fn not_regular() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "not a regular file")
}

/// The error of a file that is larger than its reader accepts.
pub fn too_large(cap: usize) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("larger than {cap} bytes"),
    )
}

/// True when `e` is the ELOOP of `O_NOFOLLOW` meeting a symlink.
pub fn is_symlink_refusal(e: &io::Error) -> bool {
    e.raw_os_error() == Some(nix::errno::Errno::ELOOP as i32)
}

/// Open `path` read-only: never blocks on a FIFO, follows a final symlink only with `Follow::Yes`, and refuses
/// anything but a regular file (checked on the descriptor).
pub fn open_regular(path: &Path, follow: Follow) -> io::Result<File> {
    let mut flags = nix::libc::O_NONBLOCK | nix::libc::O_CLOEXEC;
    if follow == Follow::No {
        flags |= nix::libc::O_NOFOLLOW;
    }
    let f = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(flags)
        .open(path)?;
    if !f.metadata()?.file_type().is_file() {
        return Err(not_regular());
    }
    Ok(f)
}

/// The whole content of an open file, refusing more than `cap` bytes (it never reads past cap + 1).
pub fn read_capped(f: &mut File, cap: usize) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let limit = u64::try_from(cap).unwrap_or(u64::MAX).saturating_add(1);
    f.take(limit).read_to_end(&mut buf)?;
    if buf.len() > cap {
        return Err(too_large(cap));
    }
    Ok(buf)
}

/// `open_regular` + `read_capped`.
pub fn read_file(path: &Path, follow: Follow, cap: usize) -> io::Result<Vec<u8>> {
    let mut f = open_regular(path, follow)?;
    read_capped(&mut f, cap)
}

/// `read_file` as UTF-8 text (invalid UTF-8 is an error, as `std::fs::read_to_string` and Python's strict decode).
pub fn read_text(path: &Path, follow: Follow, cap: usize) -> io::Result<String> {
    let bytes = read_file(path, follow, cap)?;
    String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// The lines of `text` that are at most `max_line` bytes long (longer ones are skipped, never split).
pub fn bounded_lines(text: &str, max_line: usize) -> impl Iterator<Item = &str> {
    text.lines().filter(move |l| l.len() <= max_line)
}

/// Create (or truncate) the hub's own private file for writing: never through a symlink (`O_NOFOLLOW`), never blocking
/// on a FIFO (`O_NONBLOCK`), a regular file only (fstat), mode forced to `mode` on the descriptor (O_CREAT's mode is
/// ignored for a file that already exists, possibly looser).
pub fn create_private(path: &Path, mode: u32) -> io::Result<File> {
    use std::os::unix::fs::PermissionsExt as _;
    let f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(mode)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK | nix::libc::O_CLOEXEC)
        .open(path)?;
    if !f.metadata()?.file_type().is_file() {
        return Err(not_regular());
    }
    f.set_permissions(std::fs::Permissions::from_mode(mode))?;
    Ok(f)
}

/// Refuse a path that exists but is not a regular file (a FIFO there would hang the next opener, e.g. SQLite).
/// Missing is fine; a symlink is followed, as the next opener will.
pub fn require_regular_if_present(path: &Path) -> io::Result<()> {
    match open_regular(path, Follow::Yes) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fifo(p: &Path) {
        nix::unistd::mkfifo(p, nix::sys::stat::Mode::from_bits_truncate(0o600)).unwrap();
    }

    #[test]
    fn test_a_fifo_is_refused_without_hanging() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("f");
        fifo(&p);
        let t = std::time::Instant::now();
        let e = read_file(&p, Follow::Yes, 1024).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
        assert!(t.elapsed() < std::time::Duration::from_secs(2));
        assert!(require_regular_if_present(&p).is_err());
        assert!(require_regular_if_present(&d.path().join("missing")).is_ok());
        assert!(read_file(d.path(), Follow::Yes, 1024).is_err()); // a directory
    }

    #[test]
    fn test_symlinks_follow_only_when_asked_and_the_cap_holds() {
        let d = tempfile::tempdir().unwrap();
        let real = d.path().join("real");
        std::fs::write(&real, b"0123456789").unwrap();
        let link = d.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert_eq!(read_file(&link, Follow::Yes, 64).unwrap(), b"0123456789");
        let e = read_file(&link, Follow::No, 64).unwrap_err();
        assert!(is_symlink_refusal(&e), "{e}");
        assert_eq!(read_file(&real, Follow::No, 10).unwrap().len(), 10);
        assert_eq!(
            read_file(&real, Follow::No, 9).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        let lines: Vec<&str> = bounded_lines("a\nbbbbbb\ncc", 2).collect();
        assert_eq!(lines, ["a", "cc"]);
    }

    #[test]
    fn test_create_private_refuses_symlinks_and_fifos() {
        use std::os::unix::fs::PermissionsExt as _;
        let d = tempfile::tempdir().unwrap();
        let target = d.path().join("elsewhere");
        std::fs::write(&target, b"keep").unwrap();
        let link = d.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(is_symlink_refusal(
            &create_private(&link, 0o600).unwrap_err()
        ));
        assert_eq!(std::fs::read(&target).unwrap(), b"keep");
        let f = d.path().join("fifo");
        fifo(&f);
        assert!(create_private(&f, 0o600).is_err()); // no reader: ENXIO at once, never a hang
        let p = d.path().join("ok");
        std::fs::write(&p, b"old").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
        drop(create_private(&p, 0o600).unwrap());
        let m = std::fs::metadata(&p).unwrap();
        assert_eq!((m.len(), m.permissions().mode() & 0o777), (0, 0o600));
    }
}
