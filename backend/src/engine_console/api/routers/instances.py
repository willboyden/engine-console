from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Response

from engine_console.api.deps import C, sse_response
from engine_console.domain.models import (
    CommandSnippets,
    Instance,
    InstanceCreate,
    InstancePatch,
    Page,
    PreflightReport,
)
from engine_console.services.common import sse

router = APIRouter()


@router.get("/instances")
async def list_instances(c: C, limit: int = 100, cursor: str | None = None) -> Page[Instance]:
    return c.lifecycle.list_page(limit, cursor)


@router.post("/instances", status_code=201)
async def create_instance(body: InstanceCreate, c: C) -> Instance:
    profile_params = c.profiles.get(body.profile_id).params if body.profile_id else None
    return await c.lifecycle.create(body, profile_params)


@router.post("/instances/preflight")
async def preflight(body: InstanceCreate, c: C) -> PreflightReport:
    """Dry run of the checks POST /instances performs (additive to the contract)."""
    profile_params = c.profiles.get(body.profile_id).params if body.profile_id else None
    report, _ = await c.lifecycle.preflight(body, params={**(profile_params or {}), **body.params})
    return report


@router.get("/instances/{iid}")
async def get_instance(iid: str, c: C) -> Instance:
    return c.lifecycle.get(iid)


@router.patch("/instances/{iid}")
async def patch_instance(iid: str, body: InstancePatch, c: C) -> Instance:
    return c.lifecycle.patch(iid, body)


@router.post("/instances/{iid}/stop")
async def stop(iid: str, c: C) -> Instance:
    return await c.lifecycle.stop(iid)


@router.post("/instances/{iid}/start")
async def start(iid: str, c: C) -> Instance:
    return await c.lifecycle.start(iid)


@router.post("/instances/{iid}/restart")
async def restart(iid: str, c: C) -> Instance:
    return await c.lifecycle.restart(iid)


@router.delete("/instances/{iid}", status_code=204)
async def remove(iid: str, c: C) -> Response:
    await c.lifecycle.remove(iid)
    return Response(status_code=204)


@router.get("/instances/{iid}/logs")
async def logs(iid: str, c: C, tail: int = 200) -> Response:
    return Response(await c.lifecycle.logs(iid, tail), media_type="text/plain; charset=utf-8")


@router.get("/instances/{iid}/logs/stream")
async def logs_stream(iid: str, c: C, tail: int = 100, once: bool = False) -> Any:
    c.lifecycle.get(iid)   # 404 before the stream starts

    async def gen() -> AsyncIterator[str]:
        if once:
            for line in (await c.lifecycle.logs(iid, tail)).splitlines():
                yield sse(line, "log")
            return
        async for line in c.lifecycle.follow_logs(iid, tail):
            yield sse(line, "log")
    return sse_response(gen(), c)


@router.get("/instances/{iid}/command")
async def command(iid: str, c: C) -> CommandSnippets:
    return c.lifecycle.command(iid)


