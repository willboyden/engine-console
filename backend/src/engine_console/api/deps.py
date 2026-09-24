from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from engine_console.container import Container
from engine_console.domain.errors import Forbidden, ProblemError
from engine_console.services.settings import Principal


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


def sse_response(gen: AsyncIterator[str], c: Container) -> StreamingResponse:
    """Wrap a stream with a connection cap (429 when full) and an idle timeout that ends silent streams."""
    if not c.sse.acquire():
        raise ProblemError("too many open streams; close one and retry", code="too_many_streams", status=429)
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            released = True
            c.sse.release()

    async def limited() -> AsyncIterator[str]:
        try:
            while True:
                try:
                    item = await asyncio.wait_for(gen.__anext__(), c.cfg.sse_idle_timeout_s)
                except (StopAsyncIteration, TimeoutError):
                    return
                yield item
        finally:
            release()
            aclose = getattr(gen, "aclose", None)
            if aclose is not None:
                await aclose()

    return StreamingResponse(limited(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                             background=BackgroundTask(release))
