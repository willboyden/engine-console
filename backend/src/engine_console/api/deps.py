from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from typing import Annotated

from fastapi import Depends, Request
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from engine_console.container import Container
from engine_console.domain.errors import Forbidden, TooManyStreams
from engine_console.services.common import SseLimiter
from engine_console.services.settings import Principal

# The caller of the request being handled, set by SecurityMiddleware for its whole duration. The stream caps count per
# key with it, and the SSE routes hand sse_response() only the container.
PRINCIPAL: ContextVar[Principal | None] = ContextVar("engine_console_principal", default=None)
NO_KEY = "(no key)"   # requests without a key (trusted loopback) share one bucket; key ids never look like this
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def get_container(request: Request) -> Container:
    c: Container = request.app.state.container
    return c


C = Annotated[Container, Depends(get_container)]


def require_admin(request: Request) -> Principal:
    p: Principal = request.scope.get("state", {}).get("principal", Principal("public", "public"))
    if p.role != "admin":
        raise Forbidden("admin role required")
    return p


Admin = Annotated[Principal, Depends(require_admin)]


def caller(request: Request) -> Principal:
    p: Principal = request.scope.get("state", {}).get("principal", Principal("public", "public"))
    return p


Caller = Annotated[Principal, Depends(caller)]
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def enforce_roles(request: Request) -> None:
    """Defence in depth behind SecurityMiddleware: attached to every router, so a mutating route is admin-only
    unless it is on the exact (method, path) viewer list, even if the middleware were bypassed or changed."""
    if request.method in SAFE_METHODS:
        return
    p: Principal = request.scope.get("state", {}).get("principal", Principal("public", "public"))
    if p.role == "admin":
        return
    from engine_console.api.security import viewer_may
    if p.role == "viewer" and viewer_may(request.method, request.url.path):
        return
    raise Forbidden("admin role required for this action")


def stream_key() -> str:
    """The bucket the current caller's streams count against: its API key id."""
    p = PRINCIPAL.get()
    return p.key_id if p is not None and p.key_id else NO_KEY


def take_stream_slot(lim: SseLimiter, what: str, code: str = "too_many_streams") -> Callable[[], None]:
    """One slot of `lim` for the current caller, or 429 `code` naming the cap that is full (this key's share first: that
    is the one the caller can do something about). Returns the release, which is safe to call more than once."""
    key = stream_key()
    if not lim.acquire(key):
        mine = lim.key_full(key)
        n = lim.per_key if mine else lim.limit
        detail = (f"this API key already has {n} {what} open (the most one key may hold); end one and retry" if mine
                  else f"{n} {what} are already open on this console (its limit); end one and retry")
        raise TooManyStreams(detail, code=code, scope="key" if mine else "total", limit=n)
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            released = True
            lim.release(key)

    return release


class _SlotStream(StreamingResponse):
    """An event stream that gives its slot back however it ends. The body's own `finally` covers a normal end, an engine
    error, the idle timeout and a cancelled read. This covers a client that left before the first byte (Starlette then
    never starts the body, and a background task does not run either) or while a write was in flight (which would leave
    the body suspended, holding its slot and its engine request, until garbage collection)."""

    def __init__(self, body: AsyncIterator[str], release: Callable[[], None]) -> None:
        super().__init__(body, media_type="text/event-stream", headers=SSE_HEADERS)
        self._release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()   # run the body's cleanup now (it tells the node to stop the engine request)


def slot_stream(body: AsyncIterator[str], release: Callable[[], None]) -> StreamingResponse:
    """Stream `body` as text/event-stream while holding a slot from take_stream_slot()."""
    async def guarded() -> AsyncIterator[str]:
        try:
            async for item in body:
                yield item
        finally:
            release()
            aclose = getattr(body, "aclose", None)
            if aclose is not None:
                await aclose()

    return _SlotStream(guarded(), release)


async def _until_idle(gen: AsyncIterator[str], idle_s: float) -> AsyncIterator[str]:
    try:
        while True:
            try:
                item = await asyncio.wait_for(gen.__anext__(), idle_s)
            except (StopAsyncIteration, TimeoutError):
                return
            yield item
    finally:
        aclose = getattr(gen, "aclose", None)
        if aclose is not None:
            await aclose()


def sse_response(gen: AsyncIterator[str], c: Container) -> StreamingResponse:
    """Wrap a stream with the connection caps (429 when the console's MAX_SSE_CONNECTIONS or this key's MAX_SSE_PER_KEY
    is full) and an idle timeout that ends silent streams."""
    release = take_stream_slot(c.sse, "live streams")
    return slot_stream(_until_idle(gen, c.cfg.sse_idle_timeout_s), release)
