from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Response

from engine_console.api.deps import C, sse_response
from engine_console.domain.errors import NotFound
from engine_console.domain.models import Download, DownloadRequest, LocalModel, Page
from engine_console.domain.repo_id import RepoId
from engine_console.services.common import sse
from engine_console.services.hf import build_model_info
from engine_console.services.store import paginate

router = APIRouter()


@router.get("/downloads")
async def list_downloads(c: C, limit: int = 50, cursor: str | None = None) -> Page[Download]:
    return c.downloads.list_page(limit, cursor)


@router.post("/downloads", status_code=201)
async def create_download(body: DownloadRequest, c: C) -> Download:
    return await c.downloads.create(body)


@router.get("/downloads/stream")
async def stream_downloads(c: C, once: bool = False) -> Any:
    """SSE: a snapshot of every download, then live progress. `once=true` ends after the snapshot."""
    async def gen() -> AsyncIterator[str]:
        yield sse([d.model_dump() for d in c.downloads.list_page(200).items], "snapshot")
        if once:
            return
        async for ev in c.bus.subscribe("downloads"):
            yield sse(ev, "progress")
    return sse_response(gen(), c)


@router.get("/downloads/{did}")
async def get_download(did: str, c: C) -> Download:
    return c.downloads.get(did)


@router.post("/downloads/{did}/pause")
async def pause(did: str, c: C) -> Download:
    return await c.downloads.pause(did)


@router.post("/downloads/{did}/resume")
async def resume(did: str, c: C) -> Download:
    return await c.downloads.resume(did)


@router.post("/downloads/{did}/cancel")
async def cancel(did: str, c: C) -> Download:
    return await c.downloads.cancel(did)


@router.delete("/downloads/{did}", status_code=204)
async def delete_download(did: str, c: C) -> Response:
    await c.downloads.delete(did)
    return Response(status_code=204)


@router.get("/models")
async def local_models(c: C, limit: int = 100, cursor: str | None = None) -> Page[LocalModel]:
    limit, off = paginate(limit, cursor)
    models = c.downloads.list_local()
    page = models[off: off + limit]
    for m in page:
        hub = c.downloads.local_hub_model(m.repo_id)
        if hub is None or not hub.config:
            continue
        info = build_model_info(hub)
        for a in c.adapters.list():
            try:
                if c.fit.assess_info(a.id, info, {}, []).verdict in ("fits", "tight"):
                    m.engines_that_fit.append(a.id)
            except Exception:  # noqa: BLE001, S112 - one odd model must not blank the whole library
                continue
    return Page[LocalModel](items=page, next_cursor=str(off + limit) if len(models) > off + limit else None)


@router.delete("/models/{repo_id:path}", status_code=204)
async def delete_model(repo_id: RepoId, c: C) -> Response:
    if not c.downloads.is_cached(repo_id):
        raise NotFound(f"model not in local cache: {repo_id}")
    c.downloads.delete_local(repo_id)
    return Response(status_code=204)


