"""Glue between the pure estimator and the world: adapter memory model + compat, per-host hardware, resident engines.

Host-aware (contract v2): the estimate is made against ONE host's GPUs and RAM. A unified-memory host (the GPU's
memory is the host's RAM) has a single pool, so the RAM the engine process itself needs is taken off the GPU
budget instead of being counted as a second, separate pool."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engine_console.adapters import AdapterRegistry
from engine_console.adapters.base import Hardware, ModelInfo
from engine_console.domain.errors import Unprocessable
from engine_console.domain.fleet import LOCAL
from engine_console.domain.models import FitReport, FitRequest, ResidentUse
from engine_console.domain.refs import ModelRef
from engine_console.services.fit import estimate_fit, estimate_host_ram
from engine_console.services.hardware import HardwareService
from engine_console.services.hf import HfService
from engine_console.services.libport import LibraryPort, NullLibrary
from engine_console.services.settings import SettingsService


class FitService:
    def __init__(self, adapters: AdapterRegistry, hf: HfService, hardware: HardwareService,
                 settings: SettingsService, resident: Callable[[str], list[ResidentUse]] = lambda host: [],
                 library: LibraryPort | None = None) -> None:
        self._adapters = adapters
        self._hf = hf
        self._hw = hardware
        self._settings = settings
        self._resident = resident
        self._library: LibraryPort = library or NullLibrary()

    def set_resident_provider(self, fn: Callable[[str], list[ResidentUse]]) -> None:
        self._resident = fn

    def set_library(self, library: LibraryPort) -> None:
        self._library = library

    def resolve_gpus(self, engine: str, info: ModelInfo, params: dict[str, Any], gpu_ids: list[int],
                     host_id: str = LOCAL) -> list[int]:
        """Explicit choice wins, then the configured default (only the indices this host has), else the `tp` GPUs
        with the most free VRAM."""
        if gpu_ids:
            return gpu_ids
        hw = self._hw.hardware(host_id, None)
        if (default := [i for i in self._settings.default_gpu_ids() if i in hw.gpu_ids]):
            return default
        tp = max(1, int(self._adapters.get(engine).memory_model(info, params).get("tp") or 1))
        best = sorted(range(len(hw.gpu_ids)), key=lambda i: -hw.gpu_free_gib[i])[:tp]
        return sorted(hw.gpu_ids[i] for i in best)

    async def info_for(self, host_id: str, ref: ModelRef, revision: str | None = None, *,
                       prefer_library: bool = True, allow_hub: bool = True) -> ModelInfo:
        """The model facts for a fit estimate: the host's own copy when it has one (exact sizes, no network), else (when
        `allow_hub`) the Hugging Face description of a repo that is not downloaded yet."""
        if prefer_library:
            local = await self._library.model_info(host_id, ref)
            if local is not None:
                return local
        if allow_hub and ref.format == "hf" and ref.repo_id:
            return await self._hf.model_info(ref.repo_id, revision)
        raise Unprocessable("no readable description of this model on that machine (no config.json, or the scan has "
                            "not seen it yet): rescan the library and retry", code="model_info_unavailable")

    async def assess(self, req: FitRequest, *, allow_hub: bool = True) -> FitReport:
        ref = req.model or ModelRef(format="hf", repo_id=req.repo_id)
        adapter = self._adapters.get(req.engine)
        if ref.format not in adapter.model_formats:
            raise Unprocessable(f"{adapter.display_name} cannot load {ref.format} models "
                                f"(it loads: {', '.join(sorted(adapter.model_formats))})", code="format_unsupported")
        # a bare repo id (the Models page's fit badge) is judged from Hugging Face, as in v1; a chosen library copy from the
        # host. Without allow_hub (a viewer) only a copy a machine holds can be judged.
        info = await self.info_for(req.host_id, ref, req.revision, prefer_library=req.model is not None or not allow_hub,
                                   allow_hub=allow_hub)
        return self.assess_info(req.engine, info, req.params, req.gpu_ids, req.concurrency, host_id=req.host_id)

    def assess_info(self, engine: str, info: ModelInfo, params: dict[str, Any], gpu_ids: list[int],
                    concurrency: int = 1, *, host_id: str = LOCAL) -> FitReport:
        adapter = self._adapters.get(engine)
        errors = adapter.validate(params)
        if errors:
            raise Unprocessable("invalid engine parameters", errors=errors)
        ids = self.resolve_gpus(engine, info, params, gpu_ids, host_id)
        hw = self._hw.hardware(host_id, ids or None)
        hw_all = self._hw.hardware(host_id, None)
        mem = {k: float(v) for k, v in adapter.memory_model(info, params).items()}
        compat = adapter.compatibility(info, params, hw)
        host_ram = estimate_host_ram(adapter.host_memory_gib(info, params, hw), self._hw.host_memory(host_id))
        budget_hw = self._unified_budget(hw, host_ram.needed_gib)
        report = estimate_fit(info, mem, budget_hw, concurrency=concurrency, resident=self._resident(host_id),
                              compat=compat, hw_all=hw_all)
        report.host_ram = host_ram
        if hw.unified_memory:
            report.notes.append("unified memory: the GPU and the host share one RAM pool, so the "
                                f"{host_ram.needed_gib:.1f} GiB the engine process needs was taken off the GPU budget")
        return report

    @staticmethod
    def _unified_budget(hw: Hardware, host_needed_gib: float) -> Hardware:
        if not hw.unified_memory or not hw.gpu_ids:
            return hw
        share = host_needed_gib / len(hw.gpu_ids)
        return hw.model_copy(update={"gpu_free_gib": [max(f - share, 0.0) for f in hw.gpu_free_gib]})
