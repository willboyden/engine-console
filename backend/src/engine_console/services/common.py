"""Small shared helpers: ids, clock, in-process pub/sub for SSE, redaction, SSE framing."""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import secrets
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any


def now() -> float:
    return time.time()


def new_id(prefix: str = "") -> str:
    return prefix + secrets.token_hex(6)


class EventBus:
    """Topic -> set of bounded queues. A slow SSE client drops events instead of stalling producers."""

    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue[Any]]] = defaultdict(set)

    def publish(self, topic: str, event: Any) -> None:
        for q in list(self._subs.get(topic, ())):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)

    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        q: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
        self._subs[topic].add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._subs[topic].discard(q)


def sse(data: Any, event: str | None = None) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"), default=str)
    head = f"event: {event}\n" if event else ""
    body = "\n".join(f"data: {line}" for line in payload.splitlines() or [""])
    return f"{head}{body}\n\n"


_SENSITIVE_KEY = re.compile(r"(token|secret|password|passwd|api[_-]?key|authorization|cookie|credential)", re.I)
_SECRET_VALUE = re.compile(r"\b(hf_[A-Za-z0-9]{8,}|ec_[A-Za-z0-9_\-]{16,}|sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._\-]{8,})")


def scrub(text: str) -> str:
    """Mask secret-looking tokens in free text (logs). No truncation, unlike redact()."""
    return _SECRET_VALUE.sub("[redacted]", text)


def redact(obj: Any, _depth: int = 0) -> Any:
    """Deep-copy with sensitive keys masked and secret-looking values scrubbed; long strings truncated."""
    if _depth > 8:
        return "[truncated]"
    if isinstance(obj, dict):
        return {k: ("[redacted]" if isinstance(k, str) and _SENSITIVE_KEY.search(k) else redact(v, _depth + 1))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v, _depth + 1) for v in obj[:100]]
    if isinstance(obj, str):
        s = scrub(obj)
        return s if len(s) <= 500 else s[:500] + f"...[{len(s) - 500} more chars]"
    return obj


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class SseLimiter:
    """Caps concurrent long-lived streams (log follow, SSE) so they can't exhaust the process."""

    def __init__(self, limit: int) -> None:
        self.limit, self.active = limit, 0

    def acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)
