"""Lifecycle scripts registered in each node's policy: list them, run an action, follow the job (contract §10.8).

Running one is admin-only (it starts or stops engines, possibly several GPUs' worth); reading is not."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Path

from engine_console.api.deps import Admin, C, sse_response
from engine_console.domain.fleet import ScriptInfo, ScriptJob, ScriptRunRequest
from engine_console.domain.models import HostId, Page
from engine_console.services.common import sse

router = APIRouter()
# names come from a node's policy: letters, digits, `_`, `.`, `-`, at most 32. Matched whole, never searched.
ScriptName = Annotated[str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,31}$")]
JobId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_\-]{1,64}$")]


@router.get("/scripts")
async def list_scripts(c: C, host: HostId | None = None) -> Page[ScriptInfo]:
    return Page[ScriptInfo](items=c.scripts.scripts(host), next_cursor=None)


@router.post("/scripts/{host_id}/{name}/{action}", status_code=202)
async def run_script(host_id: Annotated[str, Path(pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")], name: ScriptName,
                     action: ScriptName, c: C, _: Admin, body: ScriptRunRequest | None = None) -> ScriptJob:
    """Start one action of a script. The script, the action and the one optional argument are checked against the policy
    the node reported; the node checks them again against its own."""
    return await c.scripts.run(host_id, name, action, body.arg if body else None)


@router.get("/scripts/jobs/{job_id}")
async def get_job(job_id: JobId, c: C) -> ScriptJob:
    return c.scripts.job(job_id)


@router.get("/scripts/jobs/{job_id}/stream")
async def job_stream(job_id: JobId, c: C, once: bool = False) -> Any:
    """SSE: every output line as event `line` (the whole history first, then live), then one `end` with the job."""
    c.scripts.job(job_id)   # 404 before the stream starts

    async def gen() -> AsyncIterator[str]:
        async for event, data in c.scripts.stream(job_id, once=once):
            yield sse(data, event)
    return sse_response(gen(), c)
