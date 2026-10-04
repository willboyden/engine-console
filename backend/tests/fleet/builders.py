"""Realistic (already sanitized) node messages and host data for the fleet tests.

Everything is built by passing a raw message through `ecnode.proto.sanitize_push`, exactly as the hub's link does, so
the tests see the same shapes production code sees: unknown fields gone, missing ones None, strings capped."""
from __future__ import annotations

from typing import Any

from ecnode import proto

from engine_console.domain.fleet import HostConfig, LinkStatus
from engine_console.services.fleet_status import HostData

GIB = 1024**3


def hello(*, addresses: list[tuple[str, str]] | None = None, roots: list[dict[str, Any]] | None = None, hostname: str = "gpu-box-2",
          **policy: Any) -> dict[str, Any]:
    pol = {"path": "/etc/ecnode/policy.json", "mode": "monitor", "engine_images": ["vllm/vllm-openai:*"],
           "gateway_image": "nginx@sha256:00", "engine_network": "engine-console-engines", "port_range": [18000, 18099],
           "discovery_ports": [8000], "model_roots": roots or [], "scripts": [], "allow_pull": False, "allow_delete": False,
           "allow_ipc_host": True, "external_control": "none", "max_shm_gib": 64, **policy}
    out = proto.sanitize_push({
        "type": "hello", "proto": 1, "version": "0.4.0", "node_id": "abc",
        "host": {"hostname": hostname, "arch": "aarch64", "os": "Ubuntu 24.04", "kernel": "6.8", "cpu_model": "Grace",
                 "cpu_cores": 20, "cpu_threads": 20, "ram_total_bytes": 128 * GIB, "boot_time": 1.0, "python": "3.12.3"},
        "gpu": {"driver": "570.1", "cuda": "12.8", "source": "nvml"},
        "gpus": [{"index": 1, "uuid": "GPU-B", "name": "RTX B", "total_bytes": 96 * GIB, "compute_capability": "12.0"},
                 {"index": 0, "uuid": "GPU-A", "name": "RTX A", "total_bytes": 48 * GIB, "compute_capability": "8.9"}],
        "docker": {"available": True, "context": "rootless", "rootless": True, "version": "27", "nvidia_runtime": True},
        "addresses": [{"iface": "eth0", "addr": a, "prefix": 24} for a, _ in (addresses or [("10.0.0.5", "eth0")])],
        "policy": pol, "forced_cmd": True, "via_ssh": True})
    assert out is not None
    return out


def sample(**over: Any) -> dict[str, Any]:
    msg = {"type": "sample", "seq": 5, "t": 1000.0,
           "gpus": {"GPU-A": {"util_pct": 80.0, "mem_used_bytes": 40 * GIB, "mem_total_bytes": 48 * GIB, "temp_c": 61.0,
                              "power_w": 250.0, "power_limit_w": 300.0, "fan_pct": 44.0,
                              "procs": [{"pid": 11, "used_bytes": 30 * GIB, "container": "ec-engine-1", "name": "python3"},
                                        {"pid": 12, "used_bytes": 10 * GIB, "container": "other", "name": "x"},
                                        {"pid": 13, "name": "no-container"}]},
                    "GPU-B": {"util_pct": 0.0, "mem_used_bytes": 1 * GIB, "temp_c": 30.0}},
           "mem": {"total": 256 * GIB, "available": 200 * GIB, "free": 150 * GIB, "cached": 40 * GIB, "shmem": 1 * GIB,
                   "swap_total": 8 * GIB, "swap_free": 8 * GIB},
           "cpu": {"load1": 1.5, "load5": 1.0, "load15": 0.5, "util_pct": 12.0}, "containers": {}, **over}
    out = proto.sanitize_push(msg)
    assert out is not None
    return out


def disks(mounts: list[dict[str, Any]]) -> dict[str, Any]:
    out = proto.sanitize_push({"type": "disks", "t": 1.0, "mounts": mounts})
    assert out is not None
    return out


def mount(path: str, **kw: Any) -> dict[str, Any]:
    return {"mount": path, "fstype": "ext4", "size": 100 * GIB, "used": 40 * GIB, "free": 60 * GIB, **kw}


def root(rid: str, path: str, **kw: Any) -> dict[str, Any]:
    return {"id": rid, "path": path, "kind": "hf-home", "writable": True, "label": f"{rid} label", **kw}


def data(hid: str = "gpu-box-2", *, h: dict[str, Any] | None = None, s: dict[str, Any] | None = None,
         d: dict[str, Any] | None = None, inv: dict[str, Any] | None = None, state: str = "up", ssh: str | None = "ops@10.0.0.5",
         label: str | None = None, **link: Any) -> HostData:
    host = HostConfig(id=hid, label=label or hid.title(), ssh=None if hid == "local" else ssh)
    return HostData(host=host, link=LinkStatus(state=state, **link), hello=h, sample=s, disks=d, inventory=inv)  # type: ignore[arg-type]
