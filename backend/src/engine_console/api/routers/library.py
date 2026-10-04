"""The cross-machine model library (contract v2, docs/ARCHITECTURE.md 10.5 and 10.8)."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response
from pydantic import BaseModel, Field

from engine_console.api.deps import Admin, C
from engine_console.domain.fleet import HOST_ID_PATTERN, CopyRequest, LibraryFormat, LibraryModel
from engine_console.domain.models import Download, Page
from engine_console.services.store import paginate

router = APIRouter()


class RescanRequest(BaseModel):
    host_id: str | None = Field(default=None, pattern=HOST_ID_PATTERN)   # None: every machine that is connected


@router.get("/library")
async def list_library(c: C, host: Annotated[str | None, Query(pattern=HOST_ID_PATTERN)] = None,
                       format: LibraryFormat | None = None, q: Annotated[str | None, Query(max_length=200)] = None,
                       limit: int = 100, cursor: str | None = None) -> Page[LibraryModel]:
    """Every model on every machine, grouped by model. `host` keeps the models that machine holds or can load from
    shared storage. A machine that is down shows its last-known models, with a warning."""
    models = c.library.models(host=host, format=format, q=q)
    limit, off = paginate(limit, cursor)
    return Page[LibraryModel](items=models[off: off + limit],
                              next_cursor=str(off + limit) if len(models) > off + limit else None)


@router.post("/library/rescan", status_code=202)
async def rescan_library(c: C, _: Admin, body: RescanRequest | None = None) -> dict[str, list[str]]:
    """Ask the nodes to rescan their model roots now. The answer is the machines asked; changes arrive through the
    library as each scan finishes."""
    return {"hosts": await c.library.rescan(body.host_id if body else None)}


@router.delete("/library/copies/{copy_id}", status_code=204)
async def delete_copy(copy_id: str, c: C, _: Admin) -> Response:
    """Delete one copy from its machine. 409 while an engine uses it, 403 when that machine's node policy forbids
    deletes or the root is read-only, 503 when the machine is down."""
    await c.library.delete_copy(copy_id)
    return Response(status_code=204)


@router.post("/library/copies/{copy_id}/copy", status_code=201)
async def copy_copy(copy_id: str, body: CopyRequest, c: C, _: Admin) -> Download:
    """Copy a model to another machine (or another drive of the same one): a download row of kind "copy"."""
    return await c.downloads.copy(copy_id, body)
