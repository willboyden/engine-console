"""NVIDIA GPUs: NVML through pynvml when it is importable, else `nvidia-smi` CSV queries. Zero GPUs is fine.

Unified-memory GPUs (e.g. GB10) report no memory figures ("[N/A]" / NOT_SUPPORTED): they are reported with
`unified_memory: true` and None for their memory and per-process bytes, never a made-up number.
"""
from __future__ import annotations

import contextlib
import csv
import importlib
import io
import re
import threading
from typing import Any

from .util import Runner, clean_text, run

MIB = 1024 * 1024
STATIC_FIELDS = "index,uuid,name,memory.total,compute_cap,pci.bus_id,driver_version"
STATIC_FIELDS_OLD = "index,uuid,name,memory.total,pci.bus_id,driver_version"
SAMPLE_FIELDS = "uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed"
PROC_FIELDS = "pid,used_memory,gpu_uuid"
_CUDA = re.compile(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)")


def _num(s: str) -> float | None:
    s = s.strip()
    try:
        v = float(s)
    except ValueError:
        return None                      # "[N/A]", "[Not Supported]", "[Unknown Error]"
    return v if v == v and abs(v) < 1e15 else None


def _rows(text: str) -> list[list[str]]:
    return [[c.strip() for c in row] for row in csv.reader(io.StringIO(text)) if row and any(c.strip() for c in row)]


def _s(v: object) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


class Gpus:
    def __init__(self, *, use_nvml: bool = True, smi: str = "nvidia-smi", runner: Runner = run) -> None:
        self._smi = smi
        self._run = runner
        self._lock = threading.Lock()
        self._nv: Any = self._nvml() if use_nvml else None
        self._handles: dict[str, Any] = {}
        self._unified: set[str] = set()

    @staticmethod
    def _nvml() -> Any:
        try:
            nv = importlib.import_module("pynvml")
            nv.nvmlInit()
            return nv
        except Exception:         # no driver, no library, an old library: use nvidia-smi instead
            return None

    @property
    def source(self) -> str:
        return "nvml" if self._nv is not None else "nvidia-smi"

    # ── static ───────────────────────────────────────────────────────────────────────────────────────────
    def static(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """(GPU_STATIC list, {driver, cuda, source, error})."""
        with self._lock:
            if self._nv is not None:
                try:
                    return self._static_nvml()
                except Exception as e:    # NVML broke after init: fall back to nvidia-smi
                    self._nv = None
                    nvml_error = f"NVML failed ({type(e).__name__}); using nvidia-smi"
                    gpus, info = self._static_smi()
                    info["error"] = info["error"] or nvml_error
                    return gpus, info
            return self._static_smi()

    def _static_nvml(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        nv = self._nv
        out: list[dict[str, Any]] = []
        self._handles.clear()
        self._unified.clear()
        for i in range(int(nv.nvmlDeviceGetCount())):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            uuid = _s(nv.nvmlDeviceGetUUID(h))
            total: int | None
            try:
                total = int(nv.nvmlDeviceGetMemoryInfo(h).total)
            except nv.NVMLError:
                total = None
                self._unified.add(uuid)
            cc = pci = None
            with contextlib.suppress(nv.NVMLError):
                major, minor = nv.nvmlDeviceGetCudaComputeCapability(h)
                cc = f"{major}.{minor}"
            with contextlib.suppress(nv.NVMLError):
                pci = _s(nv.nvmlDeviceGetPciInfo(h).busId)
            self._handles[uuid] = h
            out.append({"index": i, "uuid": uuid, "name": clean_text(_s(nv.nvmlDeviceGetName(h)), 128),
                        "total_bytes": total, "compute_capability": cc, "unified_memory": total is None,
                        "pci_bus_id": pci})
        driver = cuda = None
        with contextlib.suppress(nv.NVMLError):
            driver = _s(nv.nvmlSystemGetDriverVersion())
            v = int(nv.nvmlSystemGetCudaDriverVersion())
            cuda = f"{v // 1000}.{(v % 1000) // 10}"
        return out, {"driver": driver, "cuda": cuda, "source": "nvml", "error": None}

    def _static_smi(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        res = self._run([self._smi, f"--query-gpu={STATIC_FIELDS}", "--format=csv,noheader,nounits"], timeout=8)
        fields = STATIC_FIELDS
        if not res.ok and "valid field" in (res.out + res.err):
            fields = STATIC_FIELDS_OLD
            res = self._run([self._smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], timeout=8)
        if res.rc == 127:
            return [], {"driver": None, "cuda": None, "source": "none",
                        "error": "no NVIDIA driver found (nvidia-smi is not on PATH)"}
        if not res.ok:
            msg = clean_text((res.err or res.out).strip() or f"nvidia-smi exited with {res.rc}", 300)
            return [], {"driver": None, "cuda": None, "source": "none", "error": msg}
        names = fields.split(",")
        out: list[dict[str, Any]] = []
        driver = None
        self._unified.clear()
        for row in _rows(res.out):
            if len(row) < len(names):
                continue
            d = dict(zip(names, row[: len(names)], strict=True))
            total = _num(d["memory.total"])
            uuid = d["uuid"]
            if total is None:
                self._unified.add(uuid)
            idx = _num(d["index"])
            out.append({"index": int(idx) if idx is not None else len(out), "uuid": uuid,
                        "name": clean_text(d["name"], 128), "total_bytes": int(total * MIB) if total else None,
                        "compute_capability": d.get("compute_cap") if _num(d.get("compute_cap", "")) else None,
                        "unified_memory": total is None, "pci_bus_id": d.get("pci.bus_id") or None})
            driver = driver or (d.get("driver_version") or None)
        cuda = None
        if out:
            head = self._run([self._smi], timeout=8)
            m = _CUDA.search(head.out) if head.ok else None
            cuda = m.group(1) if m else None
        return out, {"driver": driver, "cuda": cuda, "source": "nvidia-smi", "error": None}

    # ── live ─────────────────────────────────────────────────────────────────────────────────────────────
    def sample(self) -> tuple[dict[str, dict[str, Any]], list[tuple[str, int, int | None]]]:
        """({uuid: GPU_SAMPLE without procs}, [(uuid, pid, used_bytes)])."""
        with self._lock:
            if self._nv is not None and self._handles:
                try:
                    return self._sample_nvml()
                except Exception:   # one bad NVML read must not end the stream
                    return {}, []
            return self._sample_smi()

    def _sample_nvml(self) -> tuple[dict[str, dict[str, Any]], list[tuple[str, int, int | None]]]:
        nv = self._nv
        gpus: dict[str, dict[str, Any]] = {}
        procs: list[tuple[str, int, int | None]] = []

        def opt(fn: Any, *a: Any) -> Any:
            try:
                return fn(*a)
            except nv.NVMLError:
                return None

        for uuid, h in self._handles.items():
            mem = None if uuid in self._unified else opt(nv.nvmlDeviceGetMemoryInfo, h)
            util = opt(nv.nvmlDeviceGetUtilizationRates, h)
            power = opt(nv.nvmlDeviceGetPowerUsage, h)
            limit = opt(nv.nvmlDeviceGetEnforcedPowerLimit, h)
            gpus[uuid] = {"util_pct": float(util.gpu) if util is not None else None,
                          "mem_used_bytes": int(mem.used) if mem is not None else None,
                          "mem_total_bytes": int(mem.total) if mem is not None else None,
                          "temp_c": _f(opt(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU)),
                          "power_w": power / 1000.0 if isinstance(power, (int, float)) else None,
                          "power_limit_w": limit / 1000.0 if isinstance(limit, (int, float)) else None,
                          "fan_pct": _f(opt(nv.nvmlDeviceGetFanSpeed, h))}
            for p in opt(nv.nvmlDeviceGetComputeRunningProcesses, h) or []:
                used = getattr(p, "usedGpuMemory", None)
                procs.append((uuid, int(p.pid),
                              int(used) if isinstance(used, int) and uuid not in self._unified else None))
        return gpus, procs

    def _sample_smi(self) -> tuple[dict[str, dict[str, Any]], list[tuple[str, int, int | None]]]:
        res = self._run([self._smi, f"--query-gpu={SAMPLE_FIELDS}", "--format=csv,noheader,nounits"], timeout=10)
        gpus: dict[str, dict[str, Any]] = {}
        if not res.ok:
            return gpus, []
        for row in _rows(res.out):
            if len(row) < 8:
                continue
            uuid = row[0]
            used, total = _num(row[2]), _num(row[3])
            gpus[uuid] = {"util_pct": _num(row[1]),
                          "mem_used_bytes": int(used * MIB) if used is not None else None,
                          "mem_total_bytes": int(total * MIB) if total is not None else None,
                          "temp_c": _num(row[4]), "power_w": _num(row[5]), "power_limit_w": _num(row[6]),
                          "fan_pct": _num(row[7])}
        procs: list[tuple[str, int, int | None]] = []
        pres = self._run([self._smi, f"--query-compute-apps={PROC_FIELDS}", "--format=csv,noheader,nounits"],
                         timeout=10)
        if pres.ok:
            for row in _rows(pres.out):
                if len(row) < 3 or not row[0].isdigit():
                    continue
                used = _num(row[1])
                procs.append((row[2], int(row[0]), int(used * MIB) if used is not None else None))
        return gpus, procs


def _f(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
