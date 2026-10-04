"""Host memory and CPU from /proc, and cgroup v2 memory/CPU of a container through its init pid.

Why cgroups: an engine can hold tens of GiB of shared memory and page cache that no single process's RSS shows;
`docker stats` reads the same cgroup counters. RSS of the process tree is only the fallback (source "rss").
Roots are injectable (Paths) so tests use fake /proc and /sys/fs/cgroup trees.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .hostinfo import Paths, read_text

_CGROUP_PATH = re.compile(r"^/[A-Za-z0-9_.@:\-/]{0,400}$")
# A docker container's cgroup, as /proc/<pid>/cgroup shows it:
#   rootless, systemd driver  0::/user.slice/user-<uid>.slice/user@<uid>.service/user.slice/docker-<id>.scope
#   rootful, systemd driver   0::/system.slice/docker-<id>.scope
#   cgroupfs driver           0::/docker/<id>        (also below a user slice; cgroup v1 lines look alike)
CONTAINER_ID = re.compile(r"/docker[-/]([0-9a-f]{64})(?:\.scope)?(?=/|$)")
MAX_PROCS = 20000
MEM_KEYS = {"MemTotal": "total", "MemAvailable": "available", "MemFree": "free", "Cached": "cached",
            "Shmem": "shmem", "SwapTotal": "swap_total", "SwapFree": "swap_free"}


class Mem:
    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self._cpu_prev: tuple[int, int] | None = None

    # ── host ─────────────────────────────────────────────────────────────────────────────────────────────
    def meminfo(self) -> dict[str, Any]:
        out: dict[str, Any] = {v: None for v in MEM_KEYS.values()}
        for line in (read_text(self.paths.proc / "meminfo", 65536) or "").splitlines():
            key, _, rest = line.partition(":")
            if key in MEM_KEYS:
                parts = rest.split()
                if parts and parts[0].isdigit():
                    out[MEM_KEYS[key]] = int(parts[0]) * 1024
        return out

    def loadavg(self) -> tuple[float | None, float | None, float | None]:
        parts = (read_text(self.paths.proc / "loadavg", 1024) or "").split()
        vals: list[float | None] = []
        for p in parts[:3]:
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(None)
        while len(vals) < 3:
            vals.append(None)
        return vals[0], vals[1], vals[2]

    def cpu_util(self) -> float | None:
        """Percent busy across all CPUs since the previous call (None on the first call)."""
        line = (read_text(self.paths.proc / "stat", 8192) or "").split("\n", 1)[0]
        fields = line.split()
        if len(fields) < 5 or fields[0] != "cpu":
            return None
        try:
            nums = [int(x) for x in fields[1:9]]
        except ValueError:
            return None
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        prev, self._cpu_prev = self._cpu_prev, (total, idle)
        if prev is None or total <= prev[0]:
            return None
        busy = 1.0 - (idle - prev[1]) / (total - prev[0])
        return round(max(0.0, min(1.0, busy)) * 100.0, 1)

    # ── processes ────────────────────────────────────────────────────────────────────────────────────────
    def comm(self, pid: int) -> str | None:
        """The process name (comm). Never the command line: GPU boxes often carry API keys in argv."""
        text = read_text(self.paths.proc / str(pid) / "comm", 256)
        return text.strip()[:64] if text else None

    def container_id_of(self, pid: int) -> str | None:
        """The 64-hex docker container id a process belongs to, from /proc/<pid>/cgroup (the outermost one when
        containers are nested), or None for a process outside docker."""
        text = read_text(self.paths.proc / str(pid) / "cgroup", 65536) or ""
        for line in text.splitlines():
            path = line.split(":", 2)[-1].strip()
            m = CONTAINER_ID.search(path)
            if m:
                return m.group(1)
        return None

    # ── containers ───────────────────────────────────────────────────────────────────────────────────────
    def cgroup_dir(self, pid: int) -> Path | None:
        text = read_text(self.paths.proc / str(pid) / "cgroup", 65536)
        if text is None:
            return None
        line = next((ln for ln in text.splitlines() if ln.startswith("0::")), None)
        if line is None:
            return None
        path = line[3:].strip()
        # untrusted text: strict charset, no traversal, and the resolved dir must stay under the cgroup root
        if not _CGROUP_PATH.fullmatch(path) or any(seg == ".." for seg in path.split("/")):
            return None
        try:
            base = self.paths.cgroup.resolve()
            target = (base / path.lstrip("/")).resolve()
        except (OSError, RuntimeError):
            return None
        if target != base and base not in target.parents:
            return None
        return target if target.is_dir() else None

    def container_mem(self, pid: int) -> dict[str, Any] | None:
        """CG_MEM of the container whose init process is `pid` (cgroup v2, else summed RSS), or None."""
        if pid <= 0:
            return None
        d = self.cgroup_dir(pid)
        if d is not None:
            current = read_text(d / "memory.current", 64)
            stat_text = read_text(d / "memory.stat", 65536)
            if current is not None and current.strip().isdigit() and stat_text is not None:
                stat: dict[str, int] = {}
                for line in stat_text.splitlines():
                    k, _, v = line.partition(" ")
                    if v.strip().isdigit():
                        stat[k] = int(v)
                kernel = stat.get("kernel", stat.get("kernel_stack", 0) + stat.get("slab", 0)
                                  + stat.get("pagetables", 0) + stat.get("percpu", 0))
                return {"current": int(current.strip()), "anon": stat.get("anon", 0), "file": stat.get("file", 0),
                        "shmem": stat.get("shmem", 0), "kernel": kernel, "source": "cgroup"}
        rss = self._rss_tree(pid)
        if rss is None:
            return None
        return {"current": rss, "anon": rss, "file": 0, "shmem": 0, "kernel": 0, "source": "rss"}

    def container_cpu_usec(self, pid: int) -> int | None:
        d = self.cgroup_dir(pid) if pid > 0 else None
        if d is None:
            return None
        for line in (read_text(d / "cpu.stat", 8192) or "").splitlines():
            k, _, v = line.partition(" ")
            if k == "usage_usec" and v.strip().isdigit():
                return int(v)
        return None

    def _rss_tree(self, pid: int) -> int | None:
        """Sum of VmRSS over the process tree under pid (underestimates: misses shared memory and cache)."""
        children: dict[int, list[int]] = {}
        try:
            with os.scandir(self.paths.proc) as it:
                for i, entry in enumerate(it):
                    if i > MAX_PROCS:
                        break
                    if not entry.name.isdigit():
                        continue
                    stat = read_text(Path(entry.path) / "stat", 4096)
                    if not stat or ")" not in stat:
                        continue
                    try:
                        ppid = int(stat[stat.rindex(")") + 1:].split()[1])
                    except (ValueError, IndexError):
                        continue
                    children.setdefault(ppid, []).append(int(entry.name))
        except OSError:
            return None
        if not (self.paths.proc / str(pid)).is_dir():
            return None
        seen, stack, total_kb = {pid}, [pid], 0
        while stack:
            cur = stack.pop()
            for line in (read_text(self.paths.proc / str(cur) / "status", 65536) or "").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        total_kb += int(parts[1])
                    break
            for ch in children.get(cur, []):
                if ch not in seen:
                    seen.add(ch)
                    stack.append(ch)
        return total_kb * 1024
