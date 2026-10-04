"""Hosts: the fleet's machines, their link state, and the live stream the Hosts page follows (contract §10.8)."""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from engine_console.api.deps import Admin, C, sse_response
from engine_console.domain.fleet import HostDetail, HostSummary
from engine_console.domain.models import Page
from engine_console.services.common import EventBus, sse
from engine_console.services.fleet import FleetService

router = APIRouter()
STREAM_POLL_S = 1.0       # how often the stream looks for a change nobody announced (live GPU numbers, a retry clock)
STREAM_REFRESH_S = 5.0    # every host is re-sent at least this often, changed or not (contract: "at least every 5 s")


def get_fleet(request: Request) -> FleetService:
    """The app container's `fleet`. Read from the container directly (same object `C` hands out) so this module
    does not depend on how the Container dataclass declares the field."""
    fleet: FleetService = request.app.state.container.fleet
    return fleet


FleetDep = Annotated[FleetService, Depends(get_fleet)]


def _dump(s: HostSummary) -> str:
    return json.dumps(s.model_dump(mode="json"), separators=(",", ":"), default=str)


async def fleet_events(fleet: FleetService, bus: EventBus, *, once: bool = False, poll_s: float = STREAM_POLL_S,
                       refresh_s: float = STREAM_REFRESH_S) -> AsyncIterator[str]:
    """SSE text for `GET /fleet/stream`: event `host` (a HostSummary) for every host first, then again whenever one
    changes, and for every host at least every `refresh_s`.

    A link announces its state changes on the bus topic "fleet", which wakes this loop at once; everything else (live
    GPU numbers, a retry countdown) is found by comparing each host's summary every `poll_s`. The comparison, not the
    wake-up, is what guarantees a change is sent, so a missed wake-up costs at most one poll."""
    last: dict[str, tuple[str, float]] = {}

    def due() -> list[str]:
        now = time.monotonic()
        out: list[str] = []
        for s in fleet.summaries():
            data = _dump(s)
            prev = last.get(s.id)
            if prev is None or prev[0] != data or now - prev[1] >= refresh_s:
                last[s.id] = (data, now)
                out.append(sse(data, "host"))
        return out

    if once:
        for chunk in due():
            yield chunk
        return
    sub = bus.subscribe("fleet")
    wake: asyncio.Future[Any] = asyncio.ensure_future(sub.__anext__())   # arms the subscription on the next loop turn
    try:
        while True:
            for chunk in due():
                yield chunk
            done, _ = await asyncio.wait({wake}, timeout=poll_s)
            if done:
                with contextlib.suppress(Exception):
                    wake.result()
                wake = asyncio.ensure_future(sub.__anext__())
    finally:
        # Cancelling the task that is mid-__anext__ is what unsubscribes the generator (closing it directly would
        # fail: it is running). `asyncio.wait` returns once that task is over and never raises ITS cancellation, so
        # a cancellation aimed at this generator is not swallowed here.
        wake.cancel()
        await asyncio.wait({wake})
        if not wake.cancelled():
            wake.exception()    # retrieve any error so it is not reported as "never retrieved"


@router.get("/hosts")
async def list_hosts(fleet: FleetDep) -> Page[HostSummary]:
    return Page[HostSummary](items=fleet.summaries(), next_cursor=None)


@router.get("/hosts/{host_id}")
async def get_host(host_id: str, fleet: FleetDep) -> HostDetail:
    return await fleet.detail(host_id)


@router.post("/hosts/{host_id}/reconnect")
async def reconnect_host(host_id: str, fleet: FleetDep, _: Admin) -> HostSummary:
    return fleet.reconnect(host_id)


@router.get("/fleet/stream")
async def fleet_stream(c: C, fleet: FleetDep, once: bool = False) -> Any:
    return sse_response(fleet_events(fleet, c.bus, once=once), c)
