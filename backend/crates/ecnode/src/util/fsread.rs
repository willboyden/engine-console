//! Reading files that something else controls, without hanging or bloating the node.
//!
//! The node reads files it does not own: /proc, /sys and /etc of the machine, files in model roots, its policy.
//! Opening a FIFO for reading blocks until a writer appears, and a device can stream forever, so a plain
//! `File::open` + `read_to_end` lets one planted special file hang a push loop or a request. Every reader here:
//! opens with O_NONBLOCK (a FIFO open returns at once) and O_CLOEXEC, adds O_NOFOLLOW where the caller does not
//! follow symlinks, `fstat`s the open file and refuses anything that is not a regular file, and reads at most a
//! fixed number of bytes. /proc and /sys pseudo-files are regular files whose size is 0: the size is never trusted,
//! reads go until EOF or the cap.

use std::fs::File;
use std::io::{self, Read};
use std::os::fd::AsFd;
use std::path::Path;

use nix::fcntl::OFlag;
use nix::sys::stat::{FileStat, Mode};

/// S_ISREG
pub fn is_regular(st: &FileStat) -> bool {
    st.st_mode & nix::libc::S_IFMT == nix::libc::S_IFREG
}

/// The error a caller sees for a FIFO, socket, device or directory where a regular file was expected.
pub fn not_regular() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "not a regular file")
}

/// Open `path` read-only for reading its bytes: never blocks on a FIFO, follows a final symlink only when
/// `follow` is true (an intermediate symlinked directory is always followed, as the kernel does), and fails with
/// `not_regular()` for anything but a regular file.
pub fn open_regular(path: &Path, follow: bool) -> io::Result<File> {
    let mut flags = OFlag::O_RDONLY | OFlag::O_CLOEXEC | OFlag::O_NONBLOCK;
    if !follow {
        flags |= OFlag::O_NOFOLLOW;
    }
    let fd = nix::fcntl::open(path, flags, Mode::empty()).map_err(io::Error::from)?;
    let st = nix::sys::stat::fstat(fd.as_fd()).map_err(io::Error::from)?;
    if !is_regular(&st) {
        return Err(not_regular());
    }
    Ok(File::from(fd))
}

/// At most `limit` bytes of `r` (fewer at EOF). Interrupted reads are retried.
pub fn read_limit(r: &mut impl Read, limit: usize) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut chunk = [0u8; 65536];
    while out.len() < limit {
        let want = (limit - out.len()).min(chunk.len());
        match r.read(&mut chunk[..want]) {
            Ok(0) => break,
            Ok(n) => out.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

/// The first `limit` bytes of the regular file at `path` (see `open_regular`).
pub fn read_file(path: &Path, limit: usize, follow: bool) -> io::Result<Vec<u8>> {
    let mut f = open_regular(path, follow)?;
    read_limit(&mut f, limit)
}
