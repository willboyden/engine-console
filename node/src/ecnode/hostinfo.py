"""Static facts about this machine for `hello`: OS, kernel, CPU, RAM, boot time, node id, interface addresses.

Filesystem roots are injectable (Paths) so tests build fake /proc, /sys and /etc trees.
"""
from __future__ import annotations

import fcntl
import hashlib
import ipaddress
import json
import os
import platform
import socket
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import Runner, clean_text, run


def _default_state() -> Path:
    return Path(os.path.expanduser("~")) / ".local" / "state" / "ecnode"


@dataclass(frozen=True)
class Paths:
    proc: Path = Path("/proc")
    sys: Path = Path("/sys")
    etc: Path = Path("/etc")
    state: Path = field(default_factory=_default_state)

    @property
    def cgroup(self) -> Path:
        return self.sys / "fs" / "cgroup"


def read_text(path: Path, limit: int = 1 << 20) -> str | None:
    try:
        with open(path, "rb") as f:
            return f.read(limit).decode("utf-8", "replace")
    except OSError:
        return None


def node_id(paths: Paths) -> str:
    """First 16 hex of sha256(/etc/machine-id): stable per machine, reveals nothing about it."""
    for p in (paths.etc / "machine-id", paths.etc.parent / "var" / "lib" / "dbus" / "machine-id"):
        text = (read_text(p, 256) or "").strip()
        if text:
            return hashlib.sha256(text.encode()).hexdigest()[:16]
    return hashlib.sha256(("host:" + socket.gethostname()).encode()).hexdigest()[:16]


def os_release(paths: Paths) -> str | None:
    for p in (paths.etc / "os-release", paths.etc.parent / "usr" / "lib" / "os-release"):
        text = read_text(p, 65536)
        if not text:
            continue
        vals: dict[str, str] = {}
        for line in text.splitlines():
            k, eq, v = line.partition("=")
            if eq:
                vals[k.strip()] = v.strip().strip('"').strip("'")
        name = vals.get("PRETTY_NAME") or " ".join(x for x in (vals.get("NAME"), vals.get("VERSION")) if x)
        if name:
            return clean_text(name, 128)
    return None


def cpu_facts(paths: Paths) -> tuple[str | None, int | None, int | None]:
    """(model, physical cores, threads)."""
    text = read_text(paths.proc / "cpuinfo", 4 << 20) or ""
    model: str | None = None
    cores: set[tuple[str, str]] = set()
    phys, core = "0", None
    for line in text.splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k in ("model name", "Hardware") and model is None and v:
            model = v
        elif k == "physical id":
            phys = v
        elif k == "core id":
            core = v
            cores.add((phys, core))
    if model is None:
        dt = read_text(paths.sys / "firmware" / "devicetree" / "base" / "model", 256)
        model = dt.strip("\x00 \n") if dt else (platform.processor() or None)
    threads = os.cpu_count()
    return (clean_text(model, 128) if model else None), (len(cores) or threads), threads


def meminfo_total(paths: Paths) -> int | None:
    for line in (read_text(paths.proc / "meminfo", 65536) or "").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def boot_time(paths: Paths) -> float | None:
    for line in (read_text(paths.proc / "stat", 1 << 20) or "").splitlines():
        if line.startswith("btime "):
            v = line.split()[1]
            return float(v) if v.isdigit() else None
    return None


def host_facts(paths: Paths) -> dict[str, Any]:
    model, cores, threads = cpu_facts(paths)
    return {"hostname": clean_text(socket.gethostname(), 253), "arch": platform.machine(),
            "os": os_release(paths), "kernel": clean_text(platform.release(), 128), "cpu_model": model,
            "cpu_cores": cores, "cpu_threads": threads, "ram_total_bytes": meminfo_total(paths),
            "boot_time": boot_time(paths), "python": platform.python_version()}


_SKIP_IFACES = ("lo", "veth", "docker", "br-", "virbr", "cni", "flannel", "tap", "tun")


def addresses(runner: Runner = run, paths: Paths | None = None) -> list[dict[str, Any]]:
    """IPv4/IPv6 addresses of real interfaces (no loopback, link-local or container bridges)."""
    res = runner(["ip", "-j", "addr"], timeout=3)
    out: list[dict[str, Any]] = []
    if res.ok:
        try:
            data = json.loads(res.out)
        except ValueError:
            data = []
        for iface in data if isinstance(data, list) else []:
            name = iface.get("ifname") if isinstance(iface, dict) else None
            if not isinstance(name, str) or name.startswith(_SKIP_IFACES):
                continue
            for a in iface.get("addr_info") or []:
                if not isinstance(a, dict):
                    continue
                addr, prefix = a.get("local"), a.get("prefixlen")
                if isinstance(addr, str) and isinstance(prefix, int) and _usable(addr):
                    out.append({"iface": clean_text(name, 32), "addr": addr, "prefix": prefix})
        return out[:64]
    return _fallback_addresses(paths or Paths())[:64]


def _usable(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified)


def _fallback_addresses(paths: Paths) -> list[dict[str, Any]]:
    """Without `ip`: IPv6 from /proc/net/if_inet6, IPv4 (primary address only) through SIOCGIFADDR."""
    out: list[dict[str, Any]] = []
    for line in (read_text(paths.proc / "net" / "if_inet6", 1 << 20) or "").splitlines():
        p = line.split()
        if len(p) >= 6 and not p[5].startswith(_SKIP_IFACES):
            try:
                addr = str(ipaddress.IPv6Address(bytes.fromhex(p[0])))
                prefix = int(p[2], 16)
            except ValueError:
                continue
            if _usable(addr):
                out.append({"iface": p[5][:32], "addr": addr, "prefix": prefix})
    try:
        names = sorted(os.listdir(paths.sys / "class" / "net"))
    except OSError:
        names = []
    for name in names:
        if name.startswith(_SKIP_IFACES) or len(name) > 15:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                req = struct.pack("256s", name.encode())
                addr = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, req)[20:24])     # SIOCGIFADDR
                mask = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x891B, req)[20:24])     # SIOCGIFNETMASK
        except OSError:
            continue
        if _usable(addr):
            out.append({"iface": name, "addr": addr,
                        "prefix": ipaddress.IPv4Network(f"0.0.0.0/{mask}").prefixlen})
    return out
