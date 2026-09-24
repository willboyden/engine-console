"""Glue between the pure estimator and the world: adapter memory model + compat, live hardware, resident engines."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engine_console.adapters import AdapterRegistry
from engine_console.adapters.base import ModelInfo
from engine_console.domain.errors import Unprocessable
from engine_console.domain.models import FitReport, FitRequest, ResidentUse
from engine_console.services.fit import estimate_fit
from engine_console.services.hardware import HardwareService
from engine_console.services.hf import HfService
from engine_console.services.settings import SettingsService


class FitService:
    def __init__(self, adapters: AdapterRegistry, hf: HfService, hardware: HardwareService,
                 settings: SettingsService, resident: Callable[[], list[ResidentUse]] = lambda: []) -> None:
        self._adapters = adapters
        self._hf = hf
        self._hw = hardware
        self._settings = settings
        self._resident = resident

    def set_resident_provider(self, fn: Callable[[], list[ResidentUse]]) -> None:
        self._resident = fn

    def resolve_gpus(self, engine: str, info: ModelInfo, params: dict[str, Any], gpu_ids: list[int]) -> list[int]:
        """Explicit choice wins, then the configured default, else the `tp` GPUs with the most free VRAM."""
        if gpu_ids:
            return gpu_ids
        if (default := self._settings.default_gpu_ids()):
            return default
        hw = self._hw.hardware(None)
        tp = max(1, int(self._adapters.get(engine).memory_model(info, params).get("tp") or 1))
        best = sorted(range(len(hw.gpu_ids)), key=lambda i: -hw.gpu_free_gib[i])[:tp]
        return sorted(hw.gpu_ids[i] for i in best)

    async def assess(self, req: FitRequest) -> FitReport:
        info = await self._hf.model_info(req.repo_id, req.revision)
        return self.assess_info(req.engine, info, req.params, req.gpu_ids, req.concurrency)

    def assess_info(self, engine: str, info: ModelInfo, params: dict[str, Any], gpu_ids: list[int],
                    concurrency: int = 1) -> FitReport:
        adapter = self._adapters.get(engine)
        errors = adapter.validate(params)
        if errors:
            raise Unprocessable("invalid engine parameters", errors=errors)
        ids = self.resolve_gpus(engine, info, params, gpu_ids)
        hw = self._hw.hardware(ids or None)
        hw_all = self._hw.hardware(None)
        mem = {k: float(v) for k, v in adapter.memory_model(info, params).items()}
        compat = adapter.compatibility(info, params, hw)
        return estimate_fit(info, mem, hw, concurrency=concurrency, resident=self._resident(),
                            compat=compat, hw_all=hw_all)
