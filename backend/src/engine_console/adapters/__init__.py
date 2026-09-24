"""Adapter registry. Engines are imported lazily and a missing/broken adapter module is tolerated
(logged, skipped) so the console still boots with whatever engines are available."""
from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Iterable

from engine_console.adapters.base import EngineAdapter
from engine_console.domain.errors import NotFound

log = logging.getLogger(__name__)
DEFAULT_MODULES = ("engine_console.adapters.vllm", "engine_console.adapters.sglang")


class AdapterRegistry:
    def __init__(self, adapters: Iterable[EngineAdapter] = ()) -> None:
        self._by_id: dict[str, EngineAdapter] = {}
        for a in adapters:
            self.register(a)

    def register(self, adapter: EngineAdapter) -> None:
        self._by_id[adapter.id] = adapter

    def get(self, engine_id: str) -> EngineAdapter:
        try:
            return self._by_id[engine_id]
        except KeyError:
            raise NotFound(f"unknown engine: {engine_id}", code="unknown_engine") from None

    def list(self) -> list[EngineAdapter]:
        return list(self._by_id.values())

    def __contains__(self, engine_id: str) -> bool:
        return engine_id in self._by_id


def load_default_adapters(modules: Iterable[str] = DEFAULT_MODULES) -> AdapterRegistry:
    """Import each adapter module and instantiate its concrete EngineAdapter subclass(es).

    We discover the class by inspection instead of assuming a name, so adapter authors are free
    to call it VllmAdapter / VLLMAdapter / SglangAdapter.
    """
    reg = AdapterRegistry()
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name == name:
                log.info("adapter module %s not present; skipping", name)
            else:
                log.warning("adapter module %s failed to import (missing %s); skipping", name, e.name)
            continue
        except Exception as e:  # noqa: BLE001 - a broken adapter must not take the console down
            log.warning("adapter module %s failed to import: %s; skipping", name, type(e).__name__)
            continue
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if issubclass(cls, EngineAdapter) and cls is not EngineAdapter and not inspect.isabstract(cls) \
                    and cls.__module__ == mod.__name__:
                try:
                    reg.register(cls())
                except Exception as e:  # noqa: BLE001
                    log.warning("adapter %s could not be instantiated: %s", cls.__name__, type(e).__name__)
    return reg
