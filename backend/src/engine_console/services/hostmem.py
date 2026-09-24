"""Host RAM accounting. Read-only: parses /proc/meminfo and the container's cgroup (v2) files directly.

Why cgroups: a container's RSS is misleading. An engine can hold tens of GiB of shared memory / page cache
that no single process's VmRSS shows, and `docker stats` reads the same cgroup counters we read here.
The filesystem root is injectable so tests can build fake /proc and /sys/fs/cgroup trees."""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

from engine_console.domain.models import HostAlert, HostMemInfo, HostMemory

log = logging.getLogger(__name__)
GIB = float(1024**3)
_CGROUP_PATH = re.compile(r"^/[A-Za-z0-9_.@:\-/]{0,400}$")
MAX_PROCS = 20000


def _kb_to_gib(kb: float) -> float:
    return kb * 1024 / GIB


def parse_meminfo(text: str, now: float) -> HostMemInfo | None:
    vals: dict[str, float] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            try:
                vals[key] = float(parts[0])
            except ValueError:
                continue
    if "MemTotal" not in vals or "MemAvailable" not in vals:
        return None
    total, avail = vals["MemTotal"], vals["MemAvailable"]
    swap_total, swap_free = vals.get("SwapTotal", 0.0), vals.get("SwapFree", 0.0)
    return HostMemInfo(
        total_gib=_kb_to_gib(total), used_gib=_kb_to_gib(total - avail), available_gib=_kb_to_gib(avail),
        free_gib=_kb_to_gib(vals.get("MemFree", 0.0)), cached_gib=_kb_to_gib(vals.get("Cached", 0.0)),
        shmem_gib=_kb_to_gib(vals.get("Shmem", 0.0)), swap_total_gib=_kb_to_gib(swap_total),
        swap_used_gib=_kb_to_gib(max(swap_total - swap_free, 0.0)), updated_at=now)


def alerts_for(m: HostMemInfo) -> list[HostAlert]:
    out: list[HostAlert] = []
    if m.total_gib > 0:
        frac = m.available_gib / m.total_gib
        if frac < 0.05:
            out.append(HostAlert(level="crit", code="host_ram_low",
                                 message=f"Only {m.available_gib:.1f} GiB of {m.total_gib:.0f} GiB host RAM is available "
                                         f"({frac:.0%}): new engines may be OOM-killed."))
        elif frac < 0.10:
            out.append(HostAlert(level="warn", code="host_ram_low",
                                 message=f"Host RAM is getting low: {m.available_gib:.1f} GiB of {m.total_gib:.0f} GiB "
                                         f"available ({frac:.0%})."))
        if m.shmem_gib > 0.25 * m.total_gib:
            out.append(HostAlert(level="warn", code="shmem_high",
                                 message=f"{m.shmem_gib:.1f} GiB of host RAM is shared memory. It counts as page cache but "
                                         "cannot be reclaimed under pressure, so it is not really available."))
    if m.swap_total_gib > 0 and m.swap_used_gib > 0.5 * m.swap_total_gib:
        out.append(HostAlert(level="warn", code="swap_heavy",
                             message=f"Swap is {m.swap_used_gib / m.swap_total_gib:.0%} used "
                                     f"({m.swap_used_gib:.1f} of {m.swap_total_gib:.0f} GiB): the host is under memory pressure."))
    return out


class HostMemService:
    def __init__(self, root: Path = Path("/"), clock: Callable[[], float] = time.time) -> None:
        self._proc = root / "proc"
        self._cg = root / "sys" / "fs" / "cgroup"
        self._clock = clock

    # -- host ---------------------------------------------------------------------------------------------------
    def host(self) -> HostMemInfo | None:
        try:
            return parse_meminfo((self._proc / "meminfo").read_text(), self._clock())
        except OSError:
            return None

    # -- container ----------------------------------------------------------------------------------------------
    def container(self, pid: int | None) -> HostMemory | None:
        """Host RAM of the container whose init process is `pid`; None if nothing is readable."""
        if not pid or pid <= 0:
            return None
        return self._from_cgroup(pid) or self._from_rss(pid)

    def _cgroup_dir(self, pid: int) -> Path | None:
        try:
            text = (self._proc / str(pid) / "cgroup").read_text()
        except OSError:
            return None
        line = next((ln for ln in text.splitlines() if ln.startswith("0::")), None)
        if line is None:
            return None
        path = line[3:].strip()
        # untrusted text: strict charset, no traversal segments, and the resolved dir must stay under the cgroup root
        if not _CGROUP_PATH.fullmatch(path) or any(seg == ".." for seg in path.split("/")):
            log.warning("ignoring malformed cgroup path for pid %s", pid)
            return None
        try:
            base = self._cg.resolve()
            target = (base / path.lstrip("/")).resolve()
        except (OSError, RuntimeError):
            return None
        if target != base and base not in target.parents:
            log.warning("ignoring cgroup path escaping the cgroup root for pid %s", pid)
            return None
        return target if target.is_dir() else None

    def _from_cgroup(self, pid: int) -> HostMemory | None:
        d = self._cgroup_dir(pid)
        if d is None:
            return None
        try:
            current = int((d / "memory.current").read_text().strip())
            stat: dict[str, int] = {}
            for line in (d / "memory.stat").read_text().splitlines():
                k, _, v = line.partition(" ")
                if v.strip().isdigit():
                    stat[k] = int(v)
        except (OSError, ValueError):
            return None
        shmem, file_ = stat.get("shmem", 0), stat.get("file", 0)
        kernel = stat.get("kernel", stat.get("kernel_stack", 0) + stat.get("slab", 0) + stat.get("pagetables", 0)
                          + stat.get("percpu", 0))
        return HostMemory(total_gib=current / GIB, anon_gib=stat.get("anon", 0) / GIB,
                          cache_gib=max(file_ - shmem, 0) / GIB, shmem_gib=shmem / GIB, kernel_gib=kernel / GIB,
                          source="cgroup")

    def _from_rss(self, pid: int) -> HostMemory | None:
        """Fallback: sum VmRSS over the process tree. Underestimates (misses shared/cached memory), so it is
        labelled source="rss"."""
        try:
            children: dict[int, list[int]] = {}
            for i, entry in enumerate(self._proc.iterdir()):
                if i > MAX_PROCS:
                    break
                if not entry.name.isdigit():
                    continue
                try:
                    stat = (entry / "stat").read_text()
                    ppid = int(stat[stat.rindex(")") + 1:].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
                children.setdefault(ppid, []).append(int(entry.name))
            if not (self._proc / str(pid)).is_dir():
                return None
            seen, stack, total_kb = {pid}, [pid], 0.0
            while stack:
                cur = stack.pop()
                total_kb += self._vmrss_kb(cur)
                for ch in children.get(cur, []):
                    if ch not in seen:
                        seen.add(ch)
                        stack.append(ch)
        except OSError:
            return None
        gib = _kb_to_gib(total_kb)
        return HostMemory(total_gib=gib, anon_gib=gib, cache_gib=0.0, shmem_gib=0.0, kernel_gib=0.0, source="rss")

    def _vmrss_kb(self, pid: int) -> float:
        try:
            for line in (self._proc / str(pid) / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1])
        except (OSError, ValueError, IndexError):
            pass
        return 0.0
