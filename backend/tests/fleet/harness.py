"""Shared helpers for the fleet tests: how to start the fake node, fast link timings, and polling."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine_console.domain.fleet import HostConfig
from engine_console.services.common import EventBus
from engine_console.services.fleet import LinkTiming, NodeLink

from .fake_node import read_log

FAKE = Path(__file__).with_name("fake_node.py")
LOCAL_HOST = HostConfig(id="local", label="This machine")
REMOTE_HOST = HostConfig(id="gpu-box-2", label="GPU box 2", ssh="ops@10.0.0.5")
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}

# Short everything, so a drop, a silence or a backoff takes milliseconds. `fix_retry_s` is deliberately unlike any
# backoff value (<= 0.4) so a test can tell "needs a person" waits from ordinary ones.
FAST = LinkTiming(hello_timeout_s=5.0, ping_s=0.05, silent_s=2.0, stale_s=3.0, backoff_min_s=0.05, backoff_max_s=0.4,
                  stable_s=30.0, fix_retry_s=9.0, flood_bytes=500 * 1024 * 1024, flood_window_s=10.0,
                  flood_frames=1_000_000, exit_grace_s=1.0)


def node_argv(**scenario: Any) -> list[str]:
    return [sys.executable, str(FAKE), json.dumps(scenario)]


async def wait_until(pred: Callable[[], Any], timeout: float = 8.0, what: str = "the condition") -> Any:
    """Poll `pred` until it is truthy (and return that value); fail with `what` after `timeout` seconds."""
    end = time.monotonic() + timeout
    while True:
        v = pred()
        if v:
            return v
        if time.monotonic() > end:
            raise AssertionError(f"timed out after {timeout:g} s waiting for {what}")
        await asyncio.sleep(0.01)


@dataclass
class Rig:
    """A started NodeLink, its bus and the fake node's log."""
    link: NodeLink
    bus: EventBus
    log_path: Path
    changes: list[tuple[str, float | None]] = field(default_factory=list)   # (state, retry wait) per state change

    def log(self) -> list[dict[str, Any]]:
        return read_log(self.log_path)

    def events(self, ev: str) -> list[dict[str, Any]]:
        return [e for e in self.log() if e["ev"] == ev]

    async def up(self) -> None:
        await wait_until(lambda: self.link.link().state == "up", what="the link to come up")

    async def state(self, state: str, timeout: float = 8.0) -> None:
        """Wait for a link state. For `down` that means the SETTLED one, which carries the retry delay: the link
        also publishes a provisional `down` the moment the pipe closes, before it has reaped the process and
        worked out why it ended, and its error text is still the generic one."""
        def reached() -> bool:
            st = self.link.link()
            return st.state == state and (state != "down" or st.retry_in_s is not None)
        await wait_until(reached, timeout, f"state {state} (is {self.link.link()})")


def alive(pid: int) -> bool:
    """Is that process still running? A zombie (exited, not yet reaped) still answers signal 0, so look at its state."""
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, OSError):
        return False
