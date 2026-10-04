from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Response

from engine_console.api.deps import Admin, C, sse_response
from engine_console.domain.errors import NotFound
from engine_console.domain.fleet import LOCAL
from engine_console.domain.models import Download, DownloadRequest, LocalModel, Page
from engine_console.domain.refs import ModelRef
from engine_console.domain.repo_id import RepoId
from engine_console.services.common import sse
from engine_console.services.store import paginate

router = APIRouter()


@router.get("/downloads")
async def list_downloads(c: C, limit: int = 50, cursor: str | None = None) -> Page[Download]:
    return c.downloads.list_page(limit, cursor)


@router.post("/downloads", status_code=201)
async def create_download(body: DownloadRequest, c: C) -> Download:
    """Download a Hugging Face repository (or some of its files, `allow_patterns`) into a writable model root of any
    machine: `host_id` and `root_id` (default: that machine's first writable Hugging Face root)."""
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
    """The v1 shape, kept: the Hugging Face models of every machine, each with its `host_id` (`GET /library` is the
    full picture). `engines_that_fit` is judged against that machine's own GPUs."""
    limit, off = paginate(limit, cursor)
    models = c.library.local_models()
    page = models[off: off + limit]
    for m in page:
        info = await c.library.model_info(m.host_id, ModelRef(format="hf", repo_id=m.repo_id))
        if info is None:
            continue
        for a in c.adapters.list():
            try:
                if c.fit.assess_info(a.id, info, {}, [], host_id=m.host_id).verdict in ("fits", "tight"):
                    m.engines_that_fit.append(a.id)
            except Exception:  # noqa: BLE001, S112 - one odd model (or a machine that is down) must not blank the list
                continue
    return Page[LocalModel](items=page, next_cursor=str(off + limit) if len(models) > off + limit else None)


@router.delete("/models/{repo_id:path}", status_code=204)
async def delete_model(repo_id: RepoId, c: C, _: Admin) -> Response:
    """Remove a model from this machine's Hugging Face cache (the library's delete, with its checks: not while an engine
    uses it, and only where the node's policy allows deletes). Other machines' copies are not touched."""
    copy = c.library.find(LOCAL, ModelRef(format="hf", repo_id=repo_id), own=True)
    if copy is None:
        raise NotFound(f"model not in local cache: {repo_id}")
    await c.library.delete_copy(copy.id)
    return Response(status_code=204)
