"""Per-host hardware view, built from what each host's node pushed (hello = static facts, sample = live numbers).

The hub never probes NVML or /proc itself (contract v2). `report(host_id)` is the API shape (v1 plus `host_id`);
`hardware(host_id, gpu_ids)` is the adapter-facing view the fit estimator and the launch builders use.

Unified-memory machines (a GPU whose memory IS the host's RAM, e.g. GB10) cannot say how much VRAM is used, so
their GPU budget is taken from the host RAM numbers: one pool, counted once."""
from __future__ import annotations

from typing import Any

from engine_console.adapters.base import Hardware
from engine_console.domain.fleet import LOCAL, FleetPort, NodeClient, NodeUnavailable
from engine_console.domain.models import GpuStat, HardwareReport, HostMemInfo
from engine_console.services import nodeview as nv
from engine_console.services.hostmem import alerts_for, meminfo_from_sample

GIB = nv.GIB


class HardwareService:
    def __init__(self, fleet: FleetPort) -> None:
        self._fleet = fleet

    # -- raw ----------------------------------------------------------------------------------------------------
    def node(self, host_id: str) -> NodeClient:
        return self._fleet.node(host_id)

    def _require(self, host_id: str) -> tuple[NodeClient, dict[str, Any], dict[str, Any]]:
        """(node, hello, sample) or NodeUnavailable: numbers from a link that is gone would be stale, not live."""
        node = self.node(host_id)
        link = node.link()
        hello = node.latest("hello")
        if link.state not in ("up", "stale") or not hello:
            raise NodeUnavailable(host_id, link.error or f"the node is {link.state}: no hardware data")
        sample = node.latest("sample")
        if sample is None:
            raise NodeUnavailable(host_id, "the node has not sent its first sample yet; try again in a few seconds")
        return node, hello, sample

    def host_memory(self, host_id: str) -> HostMemInfo | None:
        node = self.node(host_id)
        if node.link().state not in ("up", "stale"):
            return None
        s = nv.sample(node)
        return meminfo_from_sample(s.get("mem"), nv.num(s.get("t")))

    # -- API shape ----------------------------------------------------------------------------------------------
    def gpus(self, host_id: str = LOCAL) -> list[GpuStat]:
        node, _hello, sample = self._require(host_id)
        mem = meminfo_from_sample(sample.get("mem"), nv.num(sample.get("t")))
        live: dict[str, Any] = sample["gpus"] if isinstance(sample.get("gpus"), dict) else {}
        out: list[GpuStat] = []
        for g in sorted(nv.static_gpus(node), key=lambda x: x.get("index") or 0):
            s: dict[str, Any] = live[g["uuid"]] if isinstance(live.get(g["uuid"]), dict) else {}
            total = nv.gib(g.get("total_bytes"))
            used = nv.gib(s.get("mem_used_bytes"))
            if g.get("unified_memory") and used is None and mem is not None:
                total = total or mem.total_gib   # the pool is the host's RAM
                used = max(total - mem.available_gib, 0.0)
            total = total if total is not None else (nv.gib(s.get("mem_total_bytes")) or 0.0)
            used = used if used is not None else 0.0
            out.append(GpuStat(
                index=int(g.get("index") or 0), uuid=str(g["uuid"]), name=str(g.get("name") or "GPU"),
                total_gib=total, free_gib=max(total - used, 0.0), used_gib=used, util_pct=nv.num(s.get("util_pct")),
                temp_c=nv.num(s.get("temp_c")), power_w=nv.num(s.get("power_w")), fan_pct=nv.num(s.get("fan_pct")),
                compute_capability=g.get("compute_capability") or None))
        return out

    def report(self, host_id: str = LOCAL) -> HardwareReport:
        _node, hello, sample = self._require(host_id)
        mem = meminfo_from_sample(sample.get("mem"), nv.num(sample.get("t")))
        gpu: dict[str, Any] = hello["gpu"] if isinstance(hello.get("gpu"), dict) else {}
        return HardwareReport(
            gpus=self.gpus(host_id), host_ram_gib=mem.total_gib if mem else (nv.gib(self._ram(hello)) or 0.0),
            host_ram_free_gib=mem.available_gib if mem else 0.0, source=str(gpu.get("source") or "none"),
            memory=mem, alerts=alerts_for(mem) if mem else [], host_id=host_id)

    @staticmethod
    def _ram(hello: dict[str, Any]) -> Any:
        host = hello.get("host")
        return host.get("ram_total_bytes") if isinstance(host, dict) else None

    # -- adapter view -------------------------------------------------------------------------------------------
    def hardware(self, host_id: str = LOCAL, gpu_ids: list[int] | None = None) -> Hardware:
        """The adapter-facing view restricted to `gpu_ids` (None or empty = all GPUs of that host)."""
        node, hello, sample = self._require(host_id)
        rep = self.gpus(host_id)
        sel = [g for g in rep if not gpu_ids or g.index in gpu_ids]
        unified = {str(g["uuid"]): bool(g.get("unified_memory")) for g in nv.static_gpus(node)}
        ccs = sorted({g.compute_capability for g in sel if g.compute_capability})
        mem = meminfo_from_sample(sample.get("mem"), nv.num(sample.get("t")))
        ram = mem.total_gib if mem else (nv.gib(self._ram(hello)) or 0.0)
        return Hardware(gpu_ids=[g.index for g in sel], gpu_uuids=[g.uuid for g in sel], gpu_names=[g.name for g in sel],
                        gpu_total_gib=[g.total_gib for g in sel], gpu_free_gib=[g.free_gib for g in sel],
                        compute_capability=ccs[-1] if ccs else "unknown", host_ram_gib=ram, host_id=host_id,
                        arch=nv.arch(node), unified_memory=any(unified.get(g.uuid, False) for g in sel))
