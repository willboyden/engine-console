"""Shared fixtures for the data side (library, downloads, copies, settings). Everything is offline: fake nodes that work
in a temp directory, a fake Hugging Face hub, an in-memory SQLite store."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from engine_console.adapters import AdapterRegistry
from engine_console.config import Settings
from engine_console.services.common import EventBus
from engine_console.services.downloads import DownloadService
from engine_console.services.hf import HfService
from engine_console.services.library import LibraryService
from engine_console.services.settings import SettingsService
from engine_console.services.store import Store

from .fakes import FakeFleet, FakeHub, FakeNode, StubAdapter


@pytest.fixture(autouse=True)
def _ignore_user_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on the developer's ~/.config/engine-console/env (or real EGRESS_*/HF_* variables)."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def make_settings(tmp: Path) -> Settings:
    return Settings(data_dir=tmp / "data", hf_cache_dir=tmp / "hf-unused", otlp_enabled=False, background_tasks=False,
                    hf_token=None, hf_token_file=tmp / "no-token", discovery_ports=[], fleet_enabled=False)


def make_adapters() -> AdapterRegistry:
    return AdapterRegistry([StubAdapter("vllm", frozenset({"hf", "gguf"})), StubAdapter("llamacpp", frozenset({"gguf", "plain"})),
                            StubAdapter("ollama", frozenset({"ollama"}))])


class World:
    """A hub with a fleet of fake nodes, the library over them and the download service."""

    def __init__(self, tmp: Path, nodes: list[FakeNode] | None = None, *, in_use: Callable[[str, str], list[str]] | None = None,
                 store: Store | None = None, page_limit: int = 200, hub: FakeHub | None = None, clock: Callable[[], float] | None = None,
                 max_queue: int = 50) -> None:
        self.tmp = tmp
        self.bus = EventBus()
        self.store = store or Store(":memory:")
        self.nodes = nodes or []
        self.fleet = FakeFleet(self.bus, self.nodes)
        self.in_use_calls: list[tuple[str, str]] = []
        self.busy: dict[tuple[str, str], list[str]] = {}

        def _in_use(host: str, path: str) -> list[str]:
            self.in_use_calls.append((host, path))
            return in_use(host, path) if in_use else list(self.busy.get((host, path), []))

        self.adapters = make_adapters()
        self.library = LibraryService(self.store, self.fleet, self.adapters, self.bus, in_use=_in_use,
                                      page_limit=page_limit, retry_delays=(0.01,))
        self.hub = hub or FakeHub()
        self.settings = SettingsService(self.store, make_settings(tmp))
        kw: dict[str, Any] = {"clock": clock} if clock else {}
        self.downloads = DownloadService(self.store, self.hub, HfService(self.hub, cache_ttl_s=0), self.settings, self.bus,
                                         self.fleet, self.library, concurrency=2, max_queue=max_queue, **kw)

    async def start(self) -> World:
        await self.library.start()
        await self.library.wait_idle()
        return self

    async def settle(self) -> None:
        await self.library.wait_idle()

    async def stop(self) -> None:
        await self.library.stop()
        await self.downloads.shutdown()
        self.store.close()
