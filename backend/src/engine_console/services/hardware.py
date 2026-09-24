"""Hardware inventory. NVML first (nvidia-ml-py), nvidia-smi CSV as fallback; both behind GpuProbe."""
from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from engine_console.adapters.base import Hardware
from engine_console.domain.models import GpuStat, HardwareReport
from engine_console.domain.ports import GpuProbe

log = logging.getLogger(__name__)
GIB = 1024.0**3


class NvmlProbe:
    name = "nvml"

    def query(self) -> list[GpuStat]:
        import pynvml  # nvidia-ml-py; imported lazily so machines without a driver can still import us

        pynvml.nvmlInit()
        try:
            out: list[GpuStat] = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)

                def opt(fn: Callable[..., Any], *a: Any) -> Any:  # sensors are absent on some SKUs
                    try:
                        return fn(*a)
                    except pynvml.NVMLError:
                        return None

                power = opt(pynvml.nvmlDeviceGetPowerUsage, h)
                out.append(GpuStat(
                    index=i, uuid=_s(pynvml.nvmlDeviceGetUUID(h)), name=_s(pynvml.nvmlDeviceGetName(h)),
                    total_gib=mem.total / GIB, free_gib=mem.free / GIB, used_gib=mem.used / GIB,
                    util_pct=_f(opt(lambda hh: pynvml.nvmlDeviceGetUtilizationRates(hh).gpu, h)),
                    temp_c=_f(opt(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU)),
                    power_w=power / 1000.0 if power is not None else None,
                    fan_pct=_f(opt(pynvml.nvmlDeviceGetFanSpeed, h)),
                    compute_capability=f"{cc[0]}.{cc[1]}"))
            return out
        finally:
            pynvml.nvmlShutdown()


def _s(v: object) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _f(v: object) -> float | None:
    return float(v) if isinstance(v, int | float) else None


class NvidiaSmiProbe:
    name = "nvidia-smi"
    FIELDS = "index,uuid,name,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw,fan.speed,compute_cap"

    def query(self) -> list[GpuStat]:
        res = subprocess.run(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True)
        return parse_smi_csv(res.stdout)


def _num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None  # "[N/A]"


def parse_smi_csv(text: str) -> list[GpuStat]:
    out: list[GpuStat] = []
    for line in text.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 10:
            continue
        total, free = (_num(p[3]) or 0.0) / 1024, (_num(p[4]) or 0.0) / 1024   # MiB -> GiB
        out.append(GpuStat(index=int(p[0]), uuid=p[1], name=p[2], total_gib=total, free_gib=free,
                           used_gib=total - free, util_pct=_num(p[5]), temp_c=_num(p[6]), power_w=_num(p[7]),
                           fan_pct=_num(p[8]), compute_capability=p[9]))
    return out


def read_meminfo(path: str = "/proc/meminfo") -> tuple[float, float]:
    vals: dict[str, float] = {}
    try:
        for line in Path(path).read_text().splitlines():
            k, _, rest = line.partition(":")
            vals[k] = float(rest.split()[0]) / (1024 * 1024)   # kB -> GiB
    except (OSError, ValueError, IndexError):
        pass
    return vals.get("MemTotal", 0.0), vals.get("MemAvailable", 0.0)


class HardwareService:
    def __init__(self, probes: list[GpuProbe] | None = None) -> None:
        self._probes: list[GpuProbe] = probes if probes is not None else [NvmlProbe(), NvidiaSmiProbe()]

    def report(self) -> HardwareReport:
        gpus: list[GpuStat] = []
        source = "none"
        for probe in self._probes:
            try:
                gpus = probe.query()
                source = probe.name
                break
            except Exception as e:  # noqa: BLE001 - any probe failure means "try the next one"
                log.warning("gpu probe %s failed: %s", probe.name, type(e).__name__)
        total, avail = read_meminfo()
        return HardwareReport(gpus=gpus, host_ram_gib=total, host_ram_free_gib=avail, source=source)

    def hardware(self, gpu_ids: list[int] | None = None) -> Hardware:
        """Adapter-facing view restricted to `gpu_ids` (None = all)."""
        rep = self.report()
        sel = [g for g in rep.gpus if gpu_ids is None or not gpu_ids or g.index in gpu_ids]
        ccs = sorted({g.compute_capability for g in sel if g.compute_capability})
        return Hardware(gpu_ids=[g.index for g in sel], gpu_uuids=[g.uuid for g in sel],
                        gpu_names=[g.name for g in sel], gpu_total_gib=[g.total_gib for g in sel],
                        gpu_free_gib=[g.free_gib for g in sel],
                        compute_capability=ccs[-1] if ccs else "unknown", host_ram_gib=rep.host_ram_gib)
