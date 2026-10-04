"""Filesystems relevant to this node: `/`, the filesystem of every model root, and every network mount.

Mounts come from /proc/self/mountinfo, sizes from statvfs, the drive model from /sys/block/<disk>/device/model.
NFS/CIFS sources are split into remote_host/remote_path so the hub can tell when two machines see the same
storage. statvfs and realpath on a dead network mount can block forever, so they run through util.Guarded.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hostinfo import Paths, read_text
from .policy import Root
from .util import Guarded, clean_text

PSEUDO_FS = frozenset({
    "proc", "sysfs", "devtmpfs", "devpts", "tmpfs", "cgroup", "cgroup2", "securityfs", "debugfs", "tracefs",
    "configfs", "fusectl", "mqueue", "hugetlbfs", "pstore", "bpf", "autofs", "binfmt_misc", "efivarfs", "nsfs",
    "overlay", "squashfs", "ramfs", "rpc_pipefs", "nfsd", "selinuxfs", "fuse.portal", "fuse.gvfsd-fuse",
    "fuse.snapfuse", "fuse.lxcfs", "fuse.xdg-document-portal", "devfs", "shiftfs", "zonefs", "iso9660"})
NETWORK_FS = frozenset({"nfs", "nfs4", "cifs", "smb3", "smbfs", "ceph", "glusterfs", "fuse.glusterfs", "fuse.sshfs",
                        "lustre", "beegfs", "gpfs", "9p", "afs", "fuse.juicefs", "fuse.s3fs", "fuse.rclone",
                        "fuse.cephfs", "virtiofs", "davfs", "fuse.davfs2"})
STAT_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class MountEntry:
    mount: str
    fstype: str
    source: str
    options: tuple[str, ...]
    dev: str                  # "major:minor"


def _unescape(s: str) -> str:
    """mountinfo escapes space, tab, newline and backslash as \\ooo octal."""
    if "\\" not in s:
        return s
    out, i = [], 0
    while i < len(s):
        code = s[i + 1:i + 4]
        if s[i] == "\\" and len(code) == 3 and all(c in "01234567" for c in code):
            out.append(chr(int(code, 8)))
            i += 4
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def parse_mountinfo(text: str) -> list[MountEntry]:
    out: list[MountEntry] = []
    for line in text.splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        lf, rf = left.split(), right.split()
        if len(lf) < 6 or len(rf) < 2:
            continue
        out.append(MountEntry(mount=_unescape(lf[4]), fstype=rf[0], source=_unescape(rf[1]),
                              options=tuple(lf[5].split(",")), dev=lf[2]))
    return out


def split_remote(fstype: str, source: str) -> tuple[str | None, str | None]:
    """(remote_host, remote_path) of a network filesystem source like host:/export or //host/share."""
    if fstype.startswith(("cifs", "smb")) or source.startswith("//"):
        parts = source.lstrip("/").split("/", 1)
        return (parts[0] or None), "/" + (parts[1] if len(parts) > 1 else "")
    if source.startswith("["):
        end = source.find("]")
        if end > 0:
            return source[1:end], source[end + 1:].lstrip(":") or "/"
    host, sep, path = source.partition(":")
    if not sep:
        return None, None
    host = host.rpartition("@")[2]          # sshfs: user@host:path
    return (host or None), (path or "/")


def mount_of(path: str, mounts: Sequence[MountEntry]) -> MountEntry | None:
    """The mount a path lives on: the longest mount point that is a prefix (the later one when stacked)."""
    best: MountEntry | None = None
    for m in mounts:
        mp = m.mount
        covers = path == mp or mp == "/" or path.startswith(mp.rstrip("/") + "/")
        if covers and (best is None or len(mp) >= len(best.mount)):
            best = m
    return best


class Disks:
    def __init__(self, paths: Paths, *, statvfs: Callable[[str], os.statvfs_result] = os.statvfs,
                 guard: Guarded | None = None) -> None:
        self.paths = paths
        self._statvfs = statvfs
        self._guard = guard or Guarded()

    def mounts(self) -> list[MountEntry]:
        return parse_mountinfo(read_text(self.paths.proc / "self" / "mountinfo", 4 << 20) or "")

    def resolve(self, path: str) -> str:
        """realpath of a root (its nearest existing ancestor if it is missing), bounded in time."""
        real = self._guard.call("realpath:" + path, lambda: os.path.realpath(path), STAT_TIMEOUT_S)
        return real or os.path.normpath(path)

    def report(self, roots: Sequence[Root]) -> dict[str, Any]:
        mounts = self.mounts()
        by_mount: dict[str, list[str]] = {}
        chosen: dict[str, MountEntry] = {}
        for r in roots:
            m = mount_of(self.resolve(r.path), mounts)
            if m is not None:
                chosen[m.mount] = m
                by_mount.setdefault(m.mount, []).append(r.id)
        for m in mounts:
            if m.mount == "/" or m.fstype in NETWORK_FS:
                chosen.setdefault(m.mount, m)
        # a stacked mount point: keep the entry that is on top (the last one in mountinfo)
        top = {m.mount: m for m in mounts}
        out = [self._describe(top.get(mp, m), by_mount.get(mp, [])) for mp, m in sorted(chosen.items())]
        return {"t": round(time.time(), 3), "mounts": out[:64]}

    def _describe(self, m: MountEntry, roots: list[str]) -> dict[str, Any]:
        remote = m.fstype in NETWORK_FS
        rhost, rpath = split_remote(m.fstype, m.source) if remote else (None, None)
        size = used = free = None
        st = self._guard.call("statvfs:" + m.mount, lambda: self._statvfs(m.mount), STAT_TIMEOUT_S)
        if st is not None:
            size = st.f_blocks * st.f_frsize
            free = st.f_bavail * st.f_frsize
            used = (st.f_blocks - st.f_bfree) * st.f_frsize
        device, model = self._device(m)
        return {"mount": clean_text(m.mount, 512), "fstype": clean_text(m.fstype, 32),
                "source": clean_text(m.source, 512), "device": device, "model": model, "size": size, "used": used,
                "free": free, "readonly": "ro" in m.options, "remote": remote,
                "remote_host": clean_text(rhost, 253) if rhost else None,
                "remote_path": clean_text(rpath, 512) if rpath else None, "roots": roots}

    def _device(self, m: MountEntry) -> tuple[str | None, str | None]:
        if not m.source.startswith("/dev/"):
            return None, None
        device = clean_text(m.source[len("/dev/"):], 64)
        return device, self._model(m.dev)

    def _model(self, dev: str, depth: int = 0) -> str | None:
        """Drive model of a block device number, through partitions and (up to 3 levels of) dm/md slaves."""
        if depth > 3 or ":" not in dev:
            return None
        node = Path(os.path.realpath(self.paths.sys / "dev" / "block" / dev))
        disk = node.parent.name if (node / "partition").exists() else node.name
        text = read_text(self.paths.sys / "block" / disk / "device" / "model", 256)
        if text and text.strip():
            return clean_text(text.strip(), 128)
        try:
            slaves = sorted(os.listdir(self.paths.sys / "block" / disk / "slaves"))
        except OSError:
            return None
        for s in slaves[:4]:
            num = (read_text(self.paths.sys / "class" / "block" / s / "dev", 32) or "").strip()
            found = self._model(num, depth + 1) if num else None
            if found:
                return found
        return None
