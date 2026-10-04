"""Path rules for every (root, rel) a hub names (docs/FLEET-PROTOCOL.md §4 "Path rules").

`rel` is relative: no leading "/", no "\\", no NUL or other control character, no empty, "." or ".." segment.
The parent directory is resolved with realpath and must lie inside the root's own realpath; the directory is
then opened and its fd re-checked through /proc/self/fd, so a directory swapped for a symlink in between is
caught. Final components are opened relative to that fd with O_NOFOLLOW.
"""
from __future__ import annotations

import contextlib
import errno
import os
import re
import stat
from dataclasses import dataclass

from . import proto
from .errors import OpError
from .policy import Policy, Root
from .util import is_secret_name

PROC_SELF_FD = "/proc/self/fd"
_BAD_CHARS = re.compile(r"[\x00-\x1f\x7f\\]")
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def split_rel(rel: object, *, what: str = "path") -> list[str]:
    """The segments of a valid relative path, else OpError(bad_args)."""
    if not isinstance(rel, str) or not rel:
        raise OpError(proto.BAD_ARGS, f"the {what} is empty; name a file or directory inside the model root")
    if rel.startswith("/"):
        raise OpError(proto.BAD_ARGS, f"the {what} {rel[:200]!r} must be relative to the model root, not absolute")
    if _BAD_CHARS.search(rel):
        raise OpError(proto.BAD_ARGS, f"the {what} contains a backslash or a control character")
    parts = rel.split("/")
    for p in parts:
        if p in ("", ".", ".."):
            raise OpError(proto.BAD_ARGS, f"the {what} {rel[:200]!r} has an empty, '.' or '..' segment")
        if len(p.encode("utf-8", "surrogateescape")) > 255:
            raise OpError(proto.BAD_ARGS, f"the {what} has a segment longer than 255 bytes")
    return parts


def inside(child: str, root: str) -> bool:
    return child == root or child.startswith(root.rstrip("/") + "/")


def secret_inside(real: str, root_real: str) -> bool:
    """True when any segment of `real` below the root looks like a key or credential store, so a symlink inside
    the root cannot lead a read to a secret directory that also sits inside the root."""
    rel = os.path.relpath(real, root_real) if inside(real, root_real) else real
    return any(is_secret_name(seg) for seg in rel.split("/") if seg not in ("", "."))


def fd_path(fd: int) -> str | None:
    try:
        return os.readlink(f"{PROC_SELF_FD}/{fd}")
    except OSError:
        return None


@dataclass(frozen=True)
class RootHandle:
    root: Root
    real: str            # realpath of the root directory

    def path(self, parts: list[str]) -> str:
        return os.path.join(self.real, *parts)


def open_root(policy: Policy, root_id: object, *, writable: bool = False) -> RootHandle:
    root = policy.root(root_id)
    if root is None:
        raise OpError(proto.NOT_FOUND, f"there is no model root {str(root_id)[:32]!r} in this node's policy "
                                       f"({policy.where()})")
    if writable and not root.writable:
        raise OpError(proto.POLICY_DENIED, f"model root {root.id!r} is read-only in this node's policy; an "
                                           f"administrator can set \"writable\": true in {policy.where()}")
    real = os.path.realpath(root.path)
    if not os.path.isdir(real):
        raise OpError(proto.NOT_FOUND, f"model root {root.id!r} ({root.path}) does not exist on this machine "
                                       "(not mounted?)")
    return RootHandle(root, real)


def _check_fd(fd: int, h: RootHandle) -> None:
    where = fd_path(fd)
    if where is not None and not inside(where, h.real):
        os.close(fd)
        raise OpError(proto.POLICY_DENIED, "that path leads outside the model root")


def open_dir(h: RootHandle, parts: list[str], *, create: bool = False) -> int:
    """A directory fd for root/parts. Each existing component may be a symlink only if it resolves inside the
    root. With create=True missing directories are made (0755)."""
    if not create:
        real = os.path.realpath(h.path(parts))
        if not inside(real, h.real):
            raise OpError(proto.POLICY_DENIED, "that path resolves to somewhere outside the model root")
        try:
            fd = os.open(real, DIR_FLAGS)
        except FileNotFoundError:
            raise OpError(proto.NOT_FOUND, f"directory {'/'.join(parts)[:300]!r} does not exist in the model "
                                           "root") from None
        except NotADirectoryError:
            raise OpError(proto.BAD_ARGS, f"{'/'.join(parts)[:300]!r} is not a directory") from None
        except OSError as e:
            raise OpError(proto.IO_ERROR, f"cannot open {'/'.join(parts)[:300]!r}: {e.strerror}") from None
        _check_fd(fd, h)
        return fd
    fd = os.open(h.real, DIR_FLAGS)
    try:
        for p in parts:
            try:
                nfd = os.open(p, DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(p, 0o755, dir_fd=fd)
                nfd = os.open(p, DIR_FLAGS, dir_fd=fd)
            except OSError as e:
                if e.errno not in (errno.ELOOP, errno.ENOTDIR):
                    raise
                # a symlinked directory: allowed only when it resolves inside the root
                base = fd_path(fd) or h.real
                real = os.path.realpath(os.path.join(base, p))
                if not inside(real, h.real) or not os.path.isdir(real):
                    raise OpError(proto.POLICY_DENIED, f"{p!r} is a link to somewhere outside the model root, or "
                                                       "not a directory") from None
                nfd = os.open(real, DIR_FLAGS)
            os.close(fd)
            fd = nfd
            _check_fd(fd, h)
    except OpError:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    except OSError as e:
        with contextlib.suppress(OSError):
            os.close(fd)
        if e.errno == errno.ENOSPC:
            raise OpError(proto.NO_SPACE, "the disk is full") from None
        raise OpError(proto.IO_ERROR, f"cannot create directories for that path: {e.strerror}") from None
    return fd


def lstat_at(dfd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise OpError(proto.IO_ERROR, f"cannot look at {name[:200]!r}: {e.strerror}") from None


def resolve_link_inside(dfd: int, name: str, h: RootHandle) -> str | None:
    """Realpath of a symlink at dfd/name when it resolves inside the root, else None."""
    base = fd_path(dfd)
    if base is None:
        return None
    real = os.path.realpath(os.path.join(base, name))
    return real if inside(real, h.real) else None


def resolve_existing(h: RootHandle, parts: list[str]) -> str:
    """Realpath of root/parts (it must exist and stay inside the root), for mounts and in-use checks."""
    real = os.path.realpath(h.path(parts))
    if not inside(real, h.real):
        raise OpError(proto.POLICY_DENIED, "that path resolves to somewhere outside the model root")
    if not os.path.exists(real):
        raise OpError(proto.NOT_FOUND, f"{'/'.join(parts)[:300]!r} does not exist in model root {h.root.id!r}")
    return real


def read_inside(root_real: str, path: str, max_bytes: int) -> bytes | None:
    """Bytes of a small metadata file (config.json, refs/main, a manifest) when its realpath is inside the root,
    it is a regular file of at most max_bytes and its name does not look like a secret; else None."""
    fd = open_inside(root_real, path)
    if fd is None:
        return None
    try:
        st = os.fstat(fd)
        if st.st_size > max_bytes:
            return None
        buf = b""
        while len(buf) <= max_bytes:
            chunk = os.read(fd, max_bytes + 1 - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf if len(buf) <= max_bytes else None
    except OSError:
        return None
    finally:
        os.close(fd)


def open_inside(root_real: str, path: str) -> int | None:
    """An fd of a regular file whose realpath is inside root_real (symlinks inside the root are followed, as in
    an HF snapshot), never a file with a secret-looking name; None otherwise. Never blocks on a FIFO."""
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError):
        return None
    if not inside(real, root_real) or secret_inside(real, root_real) or is_secret_name(os.path.basename(path)):
        return None
    try:
        fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except (OSError, ValueError):
        return None
    try:
        st = os.fstat(fd)
    except OSError:
        os.close(fd)
        return None
    where = fd_path(fd)
    if not stat.S_ISREG(st.st_mode) or (where is not None and not inside(where, root_real)):
        os.close(fd)
        return None
    return fd
