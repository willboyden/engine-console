"""File operations inside policy model roots (docs/FLEET-PROTOCOL.md §4): fs.stat, fs.write, fs.read, fs.symlink,
fs.put_small and fs.delete.

Every path goes through confine.split_rel + confine.open_dir: relative, no "..", parent realpath inside the root,
final component opened with O_NOFOLLOW relative to the checked directory fd. Uploads land in `<rel>.incomplete`
and are renamed into place only after the size (and, when given, the sha256 of the WHOLE file) matched.
"""
from __future__ import annotations

import base64
import binascii
import contextlib
import errno
import fcntl
import hashlib
import itertools
import os
import re
import secrets
import stat
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from . import proto
from .confine import (
    DIR_FLAGS,
    fd_path,
    inside,
    lstat_at,
    open_dir,
    open_root,
    resolve_link_inside,
    secret_inside,
    split_rel,
)
from .errors import OpError, Stream
from .policy import Policy
from .util import is_secret_name

PARTIAL = ".incomplete"
MIN_FREE = 1 << 30                  # never fill a disk below 1 GiB free
CHUNK = 512 * 1024
PUT_SMALL_MAX = 96 * 1024
MAX_FILE = 1 << 50
_SHA = re.compile(r"[0-9a-fA-F]{64}")
_BAD_TARGET = re.compile(r"[\x00-\x1f\x7f\\]")


def _name_ok_for_partial(name: str) -> None:
    if len((name + PARTIAL).encode("utf-8", "surrogateescape")) > 255:
        raise OpError(proto.BAD_ARGS, "the file name is too long to hold an upload in progress (max 243 bytes)")


# ── fs.stat ──────────────────────────────────────────────────────────────────────────────────────────────
def fs_stat(policy: Policy, args: dict[str, Any]) -> dict[str, Any]:
    h = open_root(policy, args["root"])
    items: list[dict[str, Any]] = []
    for rel in args["paths"][:256]:
        parts = split_rel(rel)
        item: dict[str, Any] = {"rel": rel, "exists": False, "is_dir": False, "size": None, "partial": 0}
        try:
            dfd = open_dir(h, parts[:-1])
        except OpError as e:
            if e.code not in (proto.NOT_FOUND, proto.BAD_ARGS):
                raise
            items.append(item)
            continue
        try:
            name = parts[-1]
            st = lstat_at(dfd, name)
            if st is not None and stat.S_ISLNK(st.st_mode):
                real = resolve_link_inside(dfd, name, h)
                st = os.stat(real) if real and os.path.exists(real) else None
            if st is not None:
                item.update(exists=True, is_dir=stat.S_ISDIR(st.st_mode),
                            size=st.st_size if stat.S_ISREG(st.st_mode) else None)
            if len((name + PARTIAL).encode("utf-8", "surrogateescape")) <= 255:
                pst = lstat_at(dfd, name + PARTIAL)
                if pst is not None and stat.S_ISREG(pst.st_mode):
                    item["partial"] = pst.st_size
        finally:
            os.close(dfd)
        items.append(item)
    return {"items": items}


# ── fs.write ─────────────────────────────────────────────────────────────────────────────────────────────
class WriteLocks:
    """One upload per file at a time within this node (flock on the partial covers other processes)."""

    def __init__(self) -> None:
        self._held: set[str] = set()
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def hold(self, key: str) -> Iterator[None]:
        with self._lock:
            if key in self._held:
                raise OpError(proto.CONFLICT, "another upload to that file is in progress")
            self._held.add(key)
        try:
            yield
        finally:
            with self._lock:
                self._held.discard(key)


def _unlink_quiet(dfd: int, name: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(name, dir_fd=dfd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def fs_write(policy: Policy, args: dict[str, Any], stream: Stream, locks: WriteLocks) -> dict[str, Any]:
    h = open_root(policy, args["root"], writable=True)
    rel = args["rel"]
    parts = split_rel(rel)
    name = parts[-1]
    _name_ok_for_partial(name)
    size = args["size"]
    if not isinstance(size, int) or not 0 <= size <= MAX_FILE:
        raise OpError(proto.BAD_ARGS, "size must be the whole file's size in bytes")
    want = args.get("sha256")
    if want is not None and not _SHA.fullmatch(want):
        raise OpError(proto.BAD_ARGS, "sha256 must be 64 hex digits")
    want = want.lower() if want else None
    offset = args.get("offset") or 0
    if not 0 <= offset <= size:
        raise OpError(proto.BAD_ARGS, f"offset {offset} is outside the file (0-{size})")
    with locks.hold(f"{h.real}\x00{rel}"):
        dfd = open_dir(h, parts[:-1], create=True)
        pfd = -1
        try:
            st = lstat_at(dfd, name)
            if st is not None and stat.S_ISDIR(st.st_mode):
                raise OpError(proto.CONFLICT, f"{rel[:300]!r} is a directory")
            try:
                pfd = os.open(name + PARTIAL, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644, dir_fd=dfd)
            except OSError as e:
                if e.errno == errno.ELOOP:
                    raise OpError(proto.POLICY_DENIED, "the upload's partial file is a symlink; refusing") from None
                raise
            try:
                fcntl.flock(pfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise OpError(proto.CONFLICT, "another process is writing that file") from None
            pst = os.fstat(pfd)
            if not stat.S_ISREG(pst.st_mode):
                raise OpError(proto.CONFLICT, "the upload's partial file is not a regular file")
            current = pst.st_size
            if offset == 0:
                os.ftruncate(pfd, 0)
            elif offset != current:
                raise OpError(proto.CONFLICT, f"resume offset {offset} does not match the partial file "
                                              f"({current} bytes): resume from {current}, or start at 0")
            vfs = os.fstatvfs(pfd)
            free = vfs.f_bavail * vfs.f_frsize
            if free - (size - offset) < MIN_FREE:
                raise OpError(proto.NO_SPACE, f"writing {size - offset} more bytes would leave less than 1 GiB free "
                                              f"on that disk ({free} bytes free)")
            hasher = hashlib.sha256() if (want or offset == 0) else None
            if hasher is not None and offset > 0:      # sha256 is of the WHOLE file: feed the existing part first
                pos = 0
                while pos < offset:
                    stream.check()
                    chunk = os.pread(pfd, min(4 << 20, offset - pos), pos)
                    if not chunk:
                        raise OpError(proto.IO_ERROR, "the partial file shrank while it was being read")
                    hasher.update(chunk)
                    pos += len(chunk)
            os.lseek(pfd, offset, os.SEEK_SET)
            total = offset
            overflow = False
            while True:
                chunk = stream.read_upload()
                if not chunk:
                    break
                if total + len(chunk) > size:
                    overflow = True
                    break
                _write_all(pfd, chunk)
                if hasher is not None:
                    hasher.update(chunk)
                total += len(chunk)
            if overflow or total != size:
                _unlink_quiet(dfd, name + PARTIAL)
                got = "more than" if overflow else f"{total} of"
                raise OpError(proto.SIZE_MISMATCH, f"received {got} {size} bytes for {rel[:300]!r}; the partial file "
                                                   "was deleted")
            os.fsync(pfd)
            digest = hasher.hexdigest() if hasher is not None else None
            if want and digest != want:
                _unlink_quiet(dfd, name + PARTIAL)
                raise OpError(proto.HASH_MISMATCH, f"sha256 of {rel[:300]!r} does not match; the partial file was "
                                                   "deleted")
            os.replace(name + PARTIAL, name, src_dir_fd=dfd, dst_dir_fd=dfd)
            with contextlib.suppress(OSError):
                os.fsync(dfd)
            return {"size": size, "sha256": digest}
        except OSError as e:
            if e.errno in (errno.ENOSPC, errno.EDQUOT):
                raise OpError(proto.NO_SPACE, "the disk filled up during the upload; the partial file was kept for "
                                              "a resume") from None
            raise OpError(proto.IO_ERROR, f"writing {rel[:300]!r} failed: {e.strerror}") from None
        finally:
            if pfd >= 0:
                os.close(pfd)
            os.close(dfd)


# ── fs.read ──────────────────────────────────────────────────────────────────────────────────────────────
def fs_read(policy: Policy, args: dict[str, Any], stream: Stream) -> dict[str, Any]:
    """Stream <root>/<rel> from `offset` (for `length` bytes, default: to the end). Result size = bytes sent."""
    h = open_root(policy, args["root"])
    rel = args["rel"]
    parts = split_rel(rel)
    if any(is_secret_name(p) for p in parts):
        raise OpError(proto.POLICY_DENIED, "refusing to read a file that looks like a key, token or credential")
    dfd = open_dir(h, parts[:-1])
    try:
        where = fd_path(dfd)
        if where is not None and secret_inside(where, h.real):
            raise OpError(proto.POLICY_DENIED, "refusing to read from a directory that looks like a key or "
                                               "credential store")
        try:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=dfd)
        except FileNotFoundError:
            raise OpError(proto.NOT_FOUND, f"{rel[:300]!r} does not exist in model root {h.root.id!r}") from None
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise OpError(proto.BAD_ARGS, f"{rel[:300]!r} is a symbolic link; read the file it points to (in an "
                                              "HF cache: the blob)") from None
            raise OpError(proto.IO_ERROR, f"cannot open {rel[:300]!r}: {e.strerror}") from None
    finally:
        os.close(dfd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OpError(proto.BAD_ARGS, f"{rel[:300]!r} is not a regular file")
        offset = args.get("offset") or 0
        if not 0 <= offset <= st.st_size:
            raise OpError(proto.BAD_ARGS, f"offset {offset} is outside the file (0-{st.st_size})")
        length = args.get("length")
        if length is not None and length < 0:
            raise OpError(proto.BAD_ARGS, "length must not be negative")
        remaining = st.st_size - offset if length is None else min(length, st.st_size - offset)
        sent, pos = 0, offset
        while sent < remaining:
            stream.check()
            chunk = os.pread(fd, min(CHUNK, remaining - sent), pos)
            if not chunk:
                break                 # the file shrank while it was being read
            stream.send_data(chunk)
            sent += len(chunk)
            pos += len(chunk)
        return {"size": sent}
    except OSError as e:
        raise OpError(proto.IO_ERROR, f"reading {rel[:300]!r} failed: {e.strerror}") from None
    finally:
        os.close(fd)


# ── fs.symlink / fs.put_small ────────────────────────────────────────────────────────────────────────────
def _replace_into(dfd: int, name: str, make: Callable[[str], None]) -> None:
    """Create a temp entry with `make(tmp)` in dfd, then rename it over `name` (a file or link, never a dir)."""
    st = lstat_at(dfd, name)
    if st is not None and stat.S_ISDIR(st.st_mode):
        raise OpError(proto.CONFLICT, f"{name[:200]!r} is a directory")
    tmp = f".{name[:200]}.ecnode-{secrets.token_hex(6)}"
    try:
        make(tmp)
        os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
    except OSError as e:
        _unlink_quiet(dfd, tmp)
        if e.errno in (errno.ENOSPC, errno.EDQUOT):
            raise OpError(proto.NO_SPACE, "the disk is full") from None
        raise OpError(proto.IO_ERROR, f"cannot write {name[:200]!r}: {e.strerror}") from None
    with contextlib.suppress(OSError):
        os.fsync(dfd)


def fs_symlink(policy: Policy, args: dict[str, Any]) -> dict[str, Any]:
    """<root>/<link> -> relative `target`; both must resolve inside the same root (HF snapshot links)."""
    h = open_root(policy, args["root"], writable=True)
    parts = split_rel(args["link"], what="link")
    target = args["target"]
    if (not isinstance(target, str) or not target or target.startswith("/") or _BAD_TARGET.search(target)
            or any(seg in ("", ".") for seg in target.split("/"))):
        raise OpError(proto.BAD_ARGS, "the link target must be a relative path like ../../blobs/<hash>")
    segs = target.split("/")
    ups = len(segs) - len(list(itertools.dropwhile(lambda s: s == "..", segs)))
    if ".." in segs[ups:]:
        # `..` only as leading segments (HF links are ../../blobs/<hash>). After a normal segment the kernel applies
        # it to wherever that segment really leads, which a lexical check cannot see: `y -> sub/..` then `z -> y/..`
        # would point z at the root's parent.
        raise OpError(proto.POLICY_DENIED, "the link target may use '..' only at its start, like ../../blobs/<hash>")
    dfd = open_dir(h, parts[:-1], create=True)
    try:
        base = fd_path(dfd) or h.path(parts[:-1])
        # realpath of the joined string, never normpath first: the kernel resolves `..` physically
        if not inside(os.path.realpath(os.path.join(base, target)), h.real):
            raise OpError(proto.POLICY_DENIED, "the link target leads outside the model root")
        _replace_into(dfd, parts[-1], lambda tmp: os.symlink(target, tmp, dir_fd=dfd))
    finally:
        os.close(dfd)
    return {}


def fs_put_small(policy: Policy, args: dict[str, Any]) -> dict[str, Any]:
    """Write at most 96 KiB atomically (HF refs/main, small metadata)."""
    h = open_root(policy, args["root"], writable=True)
    parts = split_rel(args["rel"])
    try:
        content = base64.b64decode(args["content_b64"], validate=True)
    except (binascii.Error, ValueError):
        raise OpError(proto.BAD_ARGS, "content_b64 is not valid base64") from None
    if len(content) > PUT_SMALL_MAX:
        raise OpError(proto.BAD_ARGS, f"fs.put_small takes at most {PUT_SMALL_MAX} bytes; use fs.write")
    dfd = open_dir(h, parts[:-1], create=True)

    def make(tmp: str) -> None:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644, dir_fd=dfd)
        try:
            _write_all(fd, content)
            os.fsync(fd)
        finally:
            os.close(fd)

    try:
        _replace_into(dfd, parts[-1], make)
    finally:
        os.close(dfd)
    return {}


# ── fs.delete ────────────────────────────────────────────────────────────────────────────────────────────
MAX_DELETE_DEPTH = 64


def _count(st: os.stat_result, seen: set[tuple[int, int]]) -> int:
    """Bytes a regular file gives back, each inode once (HF snapshot links and hard links share blobs)."""
    if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) in seen:
        return 0
    seen.add((st.st_dev, st.st_ino))
    return st.st_size


def _empty_dir_at(dirfd: int, seen: set[tuple[int, int]], depth: int) -> int:
    """Remove everything below the already-open directory `dirfd`; return the bytes freed.

    Only fds and names relative to them: no path string is ever resolved again, so a component swapped for a link
    while this runs is opened with O_NOFOLLOW (and refused), never followed. Links themselves are unlinked."""
    if depth > MAX_DELETE_DEPTH:
        raise OpError(proto.BAD_ARGS, f"the tree is more than {MAX_DELETE_DEPTH} directories deep; refusing to delete")
    freed = 0
    with os.scandir(dirfd) as it:
        names = [e.name for e in it]
    for name in names:
        st = lstat_at(dirfd, name)
        if st is None:
            continue
        if stat.S_ISDIR(st.st_mode):
            sub = os.open(name, DIR_FLAGS, dir_fd=dirfd)       # O_NOFOLLOW: a directory swapped for a link fails here
            try:
                freed += _empty_dir_at(sub, seen, depth + 1)
            finally:
                os.close(sub)
            os.rmdir(name, dir_fd=dirfd)
        else:
            freed += _count(st, seen)
            os.unlink(name, dir_fd=dirfd)
    return freed


def fs_delete(policy: Policy, args: dict[str, Any], running_mounts: Callable[[], Sequence[str]]) -> dict[str, Any]:
    """Delete one file or directory tree (policy allow_delete, writable root, nothing running uses it).

    The parent directory and (for a tree) the target are opened once and pinned by fd; the checks run on the fd's
    real location and the removal goes through those fds, so a path re-pointed between the check and the delete
    (the in-use check below runs docker, which takes long enough to race) cannot move it outside the root."""
    if not policy.allow_delete:
        raise OpError(proto.POLICY_DENIED, "this node's policy does not allow deleting models (allow_delete in "
                                           f"{policy.where()})")
    h = open_root(policy, args["root"], writable=True)
    rel = args["rel"]
    parts = split_rel(rel)
    dfd = open_dir(h, parts[:-1])
    tfd: int | None = None
    try:
        name = parts[-1]
        st = lstat_at(dfd, name)
        if st is None:
            raise OpError(proto.NOT_FOUND, f"{rel[:300]!r} does not exist in model root {h.root.id!r}")
        parent = fd_path(dfd)
        if parent is None or not inside(parent, h.real):
            raise OpError(proto.POLICY_DENIED, "refusing to delete: cannot confirm where that path really is")
        if stat.S_ISDIR(st.st_mode):
            try:
                tfd = os.open(name, DIR_FLAGS, dir_fd=dfd)
            except OSError as e:
                raise OpError(proto.CONFLICT, f"{rel[:300]!r} changed while being deleted: {e.strerror}") from None
            real = fd_path(tfd) or ""
        else:
            real = os.path.join(parent, name)                  # a file or a link: removed by name in its parent
        if not real or not inside(real, h.real) or real == h.real:
            raise OpError(proto.POLICY_DENIED, "refusing to delete the model root itself or anything outside it")
        for other in policy.model_roots:
            oreal = os.path.realpath(other.path)
            if other.id != h.root.id and inside(oreal, real):
                raise OpError(proto.POLICY_DENIED, f"refusing to delete {rel[:300]!r}: it holds model root "
                                                   f"{other.id!r}")
        users = [m for m in running_mounts() if inside(os.path.realpath(m), real) or inside(m, real)]
        if users:
            raise OpError(proto.IN_USE, f"a running container mounts {rel[:300]!r}; stop it before deleting")
        seen: set[tuple[int, int]] = set()
        try:
            if tfd is not None:
                if (fd_path(tfd) or "") != real:
                    raise OpError(proto.CONFLICT, f"{rel[:300]!r} moved while being deleted; nothing was removed")
                freed = _empty_dir_at(tfd, seen, 0)
                os.rmdir(name, dir_fd=dfd)
            else:
                freed = _count(st, seen)
                os.unlink(name, dir_fd=dfd)
        except OSError as e:
            raise OpError(proto.IO_ERROR, f"deleting {rel[:300]!r} failed part-way: {e.strerror}") from None
        return {"freed": freed}
    finally:
        if tfd is not None:
            os.close(tfd)
        os.close(dfd)
