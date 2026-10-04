from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from engine_console.config import Settings
from engine_console.domain.fleet import HostConfig
from engine_console.services.common import EventBus
from engine_console.services.fleet import LinkTiming, NodeLink

from .harness import ENV, FAST, LOCAL_HOST, Rig, node_argv

TEST_TIMEOUT_S = 45.0
HERE = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Every async test in this directory gets a hard time limit: a regression in flow control or in a wait must fail
    the test with a timeout, not hang the whole run (pytest-timeout is not a dependency of this project)."""
    for item in items:
        fn = getattr(item, "obj", None)
        if HERE not in item.path.parents or not inspect.iscoroutinefunction(fn):
            continue

        def limited(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            @functools.wraps(fn)
            async def run(*args: Any, **kwargs: Any) -> Any:
                loop = asyncio.get_running_loop()
                t0 = loop.time()
                try:
                    result = await asyncio.wait_for(fn(*args, **kwargs), TEST_TIMEOUT_S)
                except TimeoutError:
                    raise AssertionError(f"{fn.__name__} did not finish within {TEST_TIMEOUT_S:g} s (hung?)") from None
                if loop.time() - t0 >= TEST_TIMEOUT_S:
                    # code under test that swallows its cancellation lets wait_for return normally after the limit:
                    # that is still a hang, and must not pass
                    raise AssertionError(f"{fn.__name__} ran past {TEST_TIMEOUT_S:g} s (it swallowed its cancellation)")
                return result
            return run

        item.obj = limited(fn)  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _ignore_user_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on the developer's ~/.config/engine-console/env (or real EGRESS_*/HF_* variables)."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Settings whose every path is inside the test's temp dir: nothing under the real home is ever touched."""
    def make(**kw: Any) -> Settings:
        base: dict[str, Any] = dict(
            data_dir=tmp_path / "data", hf_cache_dir=tmp_path / "hf", hf_token_file=tmp_path / "no-token",
            hosts_file=tmp_path / "hosts.json", hub_key_file=tmp_path / "ssh" / "hub-key", node_policy=None,
            node_command=None, fleet_enabled=True, otlp_enabled=False, background_tasks=False, discovery_ports=[],
            hf_token=None, remote_node_command="/usr/local/lib/ecnode/ecnode --stdio")
        base.update(kw)
        return Settings(**base)
    return make


@pytest.fixture
async def make_rig(tmp_path: Path) -> AsyncIterator[Callable[..., Awaitable[Rig]]]:
    """Start NodeLinks against the fake node (the real subprocess path) and stop them all afterwards."""
    started: list[NodeLink] = []

    async def make(scenario: dict[str, Any] | None = None, *, host: HostConfig = LOCAL_HOST,
                   timing: LinkTiming = FAST, argv: Any = None, env: dict[str, str] | None = None,
                   start: bool = True) -> Rig:
        n = len(started)
        log_path = tmp_path / f"node-{n}.jsonl"
        bus = EventBus()
        rig_changes: list[tuple[str, float | None]] = []
        sc = {"log": str(log_path), **(scenario or {})}
        link = NodeLink(host, argv if argv is not None else node_argv(**sc), env if env is not None else ENV, bus,
                        timing=timing)
        rig = Rig(link=link, bus=bus, log_path=log_path, changes=rig_changes)
        # (state, the wait chosen before the next attempt, or None when none is pending): lets a test read the backoff
        link._on_change = lambda: rig_changes.append(
            (link.link().state, link.last_wait_s if link._retry_at is not None else None))
        started.append(link)
        if start:
            await link.start()
        return rig

    yield make
    for link in started:
        await link.stop()
