"""Host RAM accounting: pure functions only. The hub never reads /proc or cgroups (contract v2, §10.1 decision 14):
every machine's node does that and pushes `sample` messages; this module turns those numbers into the API shapes
and applies the alert thresholds.

Why cgroups on the node side: a container's RSS is misleading. An engine can hold tens of GiB of shared memory /
page cache that no single process's VmRSS shows, and `docker stats` reads the same cgroup counters."""
from __future__ import annotations

from typing import Any

from engine_console.domain.models import HostAlert, HostMemInfo, HostMemory

GIB = float(1024**3)


def _kb_to_gib(kb: float) -> float:
    return kb * 1024 / GIB


def _num(v: Any) -> float | None:
    """A finite non-negative number from untrusted node data, else None (bool is not a number here)."""
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return float(v) if v >= 0 and v == v and v != float("inf") else None


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


def meminfo_from_sample(mem: dict[str, Any] | None, t: float | None) -> HostMemInfo | None:
    """HostMemInfo from the `mem` block of a node `sample` (bytes). None when the node could not read it."""
    if not mem:
        return None
    total, avail = _num(mem.get("total")), _num(mem.get("available"))
    if total is None or avail is None:
        return None
    swap_total, swap_free = _num(mem.get("swap_total")) or 0.0, _num(mem.get("swap_free")) or 0.0
    return HostMemInfo(
        total_gib=total / GIB, used_gib=max(total - avail, 0.0) / GIB, available_gib=avail / GIB,
        free_gib=(_num(mem.get("free")) or 0.0) / GIB, cached_gib=(_num(mem.get("cached")) or 0.0) / GIB,
        shmem_gib=(_num(mem.get("shmem")) or 0.0) / GIB, swap_total_gib=swap_total / GIB,
        swap_used_gib=max(swap_total - swap_free, 0.0) / GIB, updated_at=t if t is not None else 0.0)


def container_memory(cg: dict[str, Any] | None) -> HostMemory | None:
    """HostMemory of one container from the node's cgroup block (`current`, `anon`, `file`, `shmem`, `kernel`).

    cgroup v2 `file` includes shmem, so cache = file - shmem: nothing is counted twice."""
    if not cg:
        return None
    current = _num(cg.get("current"))
    if current is None:
        return None
    shmem, file_ = _num(cg.get("shmem")) or 0.0, _num(cg.get("file")) or 0.0
    source = "rss" if cg.get("source") == "rss" else "cgroup"
    return HostMemory(total_gib=current / GIB, anon_gib=(_num(cg.get("anon")) or 0.0) / GIB,
                      cache_gib=max(file_ - shmem, 0.0) / GIB, shmem_gib=shmem / GIB,
                      kernel_gib=(_num(cg.get("kernel")) or 0.0) / GIB, source=source)


def sum_memory(parts: list[HostMemory]) -> HostMemory | None:
    """One figure for an engine made of several containers (a script's engine and its helpers)."""
    if not parts:
        return None
    return HostMemory(total_gib=sum(p.total_gib for p in parts), anon_gib=sum(p.anon_gib for p in parts),
                      cache_gib=sum(p.cache_gib for p in parts), shmem_gib=sum(p.shmem_gib for p in parts),
                      kernel_gib=sum(p.kernel_gib for p in parts),
                      source="rss" if any(p.source == "rss" for p in parts) else "cgroup")


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
